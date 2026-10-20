"""DINOv2/v3 objective prototype on a directly trained spiking ViT.

Training-only probability normalization and losses are floating point. The
student and teacher feature extractors and hidden head activations are SNNs.
    RoPE, high-resolution adaptation and official 7B-to-small distillation are
    *not* here. An optional SNN Gram-teacher refinement hook is available.
"""

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.activation_based import functional, neuron, surrogate

from .spiking_vit import SpikingVisionTransformer


class TimeBatchNorm(nn.BatchNorm1d):
    def forward(self, x):
        t, b, c = x.shape
        return super().forward(x.reshape(t * b, c)).reshape(t, b, c)


def spike_diversity_retention(input_spikes, output_spikes,
                              positions=None, num_positions=None):
    """Penalize a spiking head only when it erases input token diversity."""
    if input_spikes.ndim != 3 or output_spikes.ndim != 3 or (
            input_spikes.shape[:2] != output_spikes.shape[:2]):
        raise ValueError('Expected aligned [time, tokens, channels] spike tensors')
    input_rate = input_spikes.float().mean(0)
    output_rate = output_spikes.float().mean(0)
    if positions is not None:
        if (positions.ndim != 1 or len(positions) != len(input_rate) or
                num_positions is None or num_positions < 1):
            raise ValueError('Position groups must align with visible tokens')
        positions = positions.to(device=input_rate.device, dtype=torch.long)
        counts = torch.bincount(positions, minlength=num_positions).float()
        denominator = counts.clamp_min(1)[:, None]
        def grouped_spread(rate):
            sums = rate.new_zeros(num_positions, rate.shape[-1]).index_add_(
                0, positions, rate)
            squares = rate.new_zeros(num_positions, rate.shape[-1]).index_add_(
                0, positions, rate.square())
            mean = sums / denominator
            variance = (squares / denominator - mean.square()).clamp_min(0)
            # A channel can have exactly zero variance for binary spikes.
            # sqrt(0) has an infinite derivative; even a zero ReLU penalty
            # can then send NaNs through the shared backbone.
            return (variance + 1e-6).sqrt().mean(-1)
        valid = counts >= 2
        if not bool(valid.any()):
            return output_rate.sum() * 0
        input_spread = grouped_spread(input_rate).detach()
        output_spread = grouped_spread(output_rate)
        return F.relu(input_spread[valid] - output_spread[valid]).mean()
    input_spread = input_rate.std(0, unbiased=False).mean().detach()
    output_spread = output_rate.std(0, unbiased=False).mean()
    return F.relu(input_spread - output_spread)


class SpikingProjectionHead(nn.Module):
    """Spike-communicating MLP; continuous prototypes are loss readouts."""

    def __init__(self, dim, hidden=2048, bottleneck=256, prototypes=65536,
                 layers=3, prototype_readout='linear', shortcut='none'):
        super().__init__()
        if prototype_readout not in ('linear', 'cosine'):
            raise ValueError('Unknown prototype readout')
        self.prototype_readout = prototype_readout
        if shortcut not in ('none', 'input_or'):
            raise ValueError('Unknown spiking projection shortcut')
        if shortcut == 'input_or' and bottleneck > dim:
            raise ValueError('Input spike shortcut requires bottleneck <= input width')
        self.shortcut = shortcut
        self.bottleneck = bottleneck
        if layers not in (2, 3):
            raise ValueError('Projection MLP must have two or three linear layers')
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.bn1 = TimeBatchNorm(hidden)
        self.lif1 = neuron.LIFNode(tau=2., v_threshold=.5, v_reset=0.,
                                   detach_reset=True, surrogate_function=surrogate.ATan(),
                                   backend='torch', step_mode='m')
        if layers == 3:
            self.fc_mid = nn.Linear(hidden, hidden, bias=False)
            self.bn_mid = TimeBatchNorm(hidden)
            self.lif_mid = neuron.LIFNode(tau=2., v_threshold=.5, v_reset=0.,
                                           detach_reset=True,
                                           surrogate_function=surrogate.ATan(),
                                           backend='torch', step_mode='m')
        else:
            self.fc_mid = self.bn_mid = self.lif_mid = None
        self.fc2 = nn.Linear(hidden, bottleneck, bias=False)
        self.bn2 = TimeBatchNorm(bottleneck)
        self.lif2 = neuron.LIFNode(tau=2., v_threshold=.5, v_reset=0.,
                                   detach_reset=True, surrogate_function=surrogate.ATan(),
                                   backend='torch', step_mode='m')
        self.prototypes = nn.Parameter(torch.randn(prototypes, bottleneck) * .02)

    def forward(self, spikes, return_diversity=False, prediction_count=None,
                diversity_positions=None, diversity_num_positions=None):
        if prediction_count is None:
            prediction_count = spikes.shape[1]
        if not 0 < prediction_count <= spikes.shape[1]:
            raise ValueError('Invalid number of prediction tokens')
        h = self.lif1(self.bn1(self.fc1(spikes)))
        if self.fc_mid is not None:
            h = self.lif_mid(self.bn_mid(self.fc_mid(h)))
        h = self.lif2(self.bn2(self.fc2(h)))
        if self.shortcut == 'input_or':
            # A binary skip keeps observed patch events represented even if
            # the deep spiking MLP loses sample diversity during training.
            h = torch.maximum(h, spikes[..., :self.bottleneck])
        # Normalization and the prototype classifier are continuous training
        # readouts, not a hidden ANN layer between SNN backbone blocks.
        feature = F.normalize(h[:, :prediction_count].float().mean(0),
                              dim=-1, eps=1e-6)
        weight = (F.normalize(self.prototypes, dim=-1, eps=1e-6)
                  if self.prototype_readout == 'cosine' else self.prototypes)
        scores = F.linear(feature, weight)
        if return_diversity:
            diversity_start = (prediction_count if prediction_count < spikes.shape[1]
                               else 0)
            return scores, spike_diversity_retention(
                spikes[:, diversity_start:], h[:, diversity_start:],
                diversity_positions, diversity_num_positions)
        return scores


class SpikingDinoEncoder(nn.Module):
    def __init__(self, image_size=128, patch_size=16, dim=384, depth=12,
                 heads=6, head_hidden=2048, bottleneck=256, prototypes=65536,
                 head_layers=3, prototype_readout='linear',
                 attention_variant='matrix_recode',
                 cls_aggregation='attention_only',
                 residual_policy='accumulate',
                 dino_shortcut='none',
                 ibot_shortcut='none',
                 masked_context_bridge='none',
                 residual_gradient_policy='full',
                 block_variant='standard',
                 dino_hidden=None, dino_bottleneck=None, dino_prototypes=None,
                 ibot_hidden=None, ibot_bottleneck=None, ibot_prototypes=None):
        super().__init__()
        self.backbone = SpikingVisionTransformer(image_size=image_size,
                                                 patch_size=patch_size,
                                                 dim=dim, depth=depth, heads=heads,
                                                 attention_variant=attention_variant,
                                                 cls_aggregation=cls_aggregation,
                                                 residual_policy=residual_policy,
                                                 masked_context_bridge=masked_context_bridge,
                                                 residual_gradient_policy=residual_gradient_policy,
                                                 block_variant=block_variant)
        self.dino_head = SpikingProjectionHead(
            dim, dino_hidden or head_hidden, dino_bottleneck or bottleneck,
            dino_prototypes or prototypes, head_layers, prototype_readout,
            dino_shortcut)
        self.ibot_head = SpikingProjectionHead(
            dim, ibot_hidden or head_hidden, ibot_bottleneck or bottleneck,
            ibot_prototypes or prototypes, head_layers, prototype_readout,
            ibot_shortcut)

    def forward(self, events, mask=None, patch_head_mask=None,
                compute_patch_head=True, patch_diversity=False,
                patch_diversity_source='masked', compute_cls_head=True):
        cls, patch = self.backbone(events, mask)
        t, b, p, d = patch.shape
        cls_scores = self.dino_head(cls) if compute_cls_head else None
        patch_scores = None
        patch_diversity_gap = None
        if compute_patch_head:
            if patch_head_mask is not None:
                if patch_head_mask.shape != (b, p) or patch_head_mask.dtype != torch.bool:
                    raise ValueError('patch_head_mask must index the current patch grid')
                patch_input = patch[:, patch_head_mask]
            else:
                patch_input = patch.reshape(t, b * p, d)
            if patch_input.shape[1]:
                if patch_diversity:
                    if patch_diversity_source in ('visible', 'visible_position') and patch_head_mask is not None:
                        # The auxiliary spread constraint diagnoses the head;
                        # keep it from injecting a second, potentially huge
                        # gradient into the shared spiking backbone.
                        visible = patch[:, ~patch_head_mask].detach()
                        visible_positions = ((~patch_head_mask).nonzero(
                            as_tuple=False)[:, 1] if
                            patch_diversity_source == 'visible_position' else None)
                        combined = torch.cat((patch_input, visible), dim=1)
                        patch_scores, patch_diversity_gap = self.ibot_head(
                            combined, return_diversity=True,
                            prediction_count=patch_input.shape[1],
                            diversity_positions=visible_positions,
                            diversity_num_positions=patch_head_mask.shape[1]
                            if visible_positions is not None else None)
                    else:
                        patch_scores, patch_diversity_gap = self.ibot_head(
                            patch_input, return_diversity=True)
                else:
                    patch_scores = self.ibot_head(patch_input)
            else:
                patch_scores = patch.new_empty((0, self.ibot_head.prototypes.shape[0]))
        return {'cls_spikes': cls, 'patch_spikes': patch,
                'cls_scores': cls_scores, 'patch_scores': patch_scores,
                'patch_diversity_gap': patch_diversity_gap}


@torch.no_grad()
def sinkhorn_knopp(logits, temperature=.04, iterations=3):
    """Log-domain Sinkhorn; same assignments without large-logit underflow."""

    if logits.ndim != 2 or not logits.shape[0] or temperature <= 0:
        raise ValueError('Expected a nonempty [items, prototypes] matrix')
    q = (logits.float() / temperature).T
    q = q - torch.logsumexp(q.reshape(-1), dim=0)
    k, b = q.shape
    for _ in range(iterations):
        q = q - torch.logsumexp(q, dim=1, keepdim=True) - math.log(k)
        q = q - torch.logsumexp(q, dim=0, keepdim=True) - math.log(b)
    return (q + math.log(b)).exp().T


def masked_patch_cross_entropy(logits, targets, masks, temperature=.1):
    """Mean within each masked sample, then over ALL global sample-crops.

    Unmasked sample-crops contribute zero but remain in the denominator,
    matching DINOv3 forward_masked. Boolean indexing is crop-major.
    """
    mask = torch.cat(masks, dim=0)
    if mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError('Expected boolean [global sample-crops, patches] masks')
    if logits.shape != targets.shape or logits.shape[0] != int(mask.sum()):
        raise ValueError('Masked logits and targets do not match selected patches')
    weights = mask.sum(-1).clamp_min(1).float().reciprocal()
    weights = weights[:, None].expand_as(mask)[mask]
    ce = -(targets.detach() * F.log_softmax(logits.float() / temperature, -1)).sum(-1)
    return (ce * weights).sum() / mask.shape[0]


def koleo_loss(cls_rate):
    if cls_rate.shape[0] < 2:
        raise ValueError('KoLeo needs at least two samples')
    features = F.normalize(cls_rate.float(), dim=-1, eps=1e-6)
    distances = torch.cdist(features, features, p=2)
    distances = distances.masked_fill(torch.eye(features.shape[0],
                                                device=features.device,
                                                dtype=torch.bool), float('inf'))
    return -distances.min(dim=1).values.clamp_min(1e-6).log().mean()


def gram_loss(student_patch_rate, teacher_patch_rate):
    """Compare intra-view patch geometry, never cross-view patch indices."""

    if student_patch_rate.shape != teacher_patch_rate.shape:
        raise ValueError('Gram patch grids must align')
    s = F.normalize(student_patch_rate.float(), dim=-1, eps=1e-6)
    g = F.normalize(teacher_patch_rate.detach().float(), dim=-1, eps=1e-6)
    return ((s @ s.transpose(-1, -2) - g @ g.transpose(-1, -2)) ** 2).mean()


def _independent(model, events, mask=None, patch_head_mask=None,
                 compute_patch_head=True, patch_diversity=False,
                 patch_diversity_source='masked', compute_cls_head=True):
    functional.reset_net(model)
    try:
        return model(events, mask, patch_head_mask, compute_patch_head,
                     patch_diversity, patch_diversity_source,
                     compute_cls_head)
    finally:
        functional.reset_net(model)


def _use_current_batch_stats(module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None
        module.num_batches_tracked = None


class ChannelLayerNorm2d(nn.Module):
    """Normalize each patch position over channels, without batch statistics."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _replace_bn_with_layer_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, ChannelLayerNorm2d(child.num_features))
        elif isinstance(child, nn.BatchNorm1d):
            setattr(module, name, nn.LayerNorm(child.num_features))
        else:
            _replace_bn_with_layer_norm(child)


class SpikingDinoV3Prototype(nn.Module):
    """EMA SNN teacher, cross-view DINO, same-view masked iBOT, KoLeo.

    The first two views must be same-resolution global crops. Their masks
    index the corresponding teacher patch grid, never a different crop.
    """

    def __init__(self, *, student_temp=.1, teacher_temp=.04,
                 teacher_momentum=.996, koleo_weight=.1, gram_weight=1.,
                 balance_mode='joint', norm_policy='ema_bn', ibot_reduction='sample',
                 target_policy='sinkhorn', center_momentum=.9,
                 patch_teacher_aggregation='single_view',
                 ibot_diversity_weight=0.,
                 ibot_diversity_source='masked',
                 global_cls_policy='masked',
                 ibot_residual_gradient_policy='inherit',
                 **encoder_kwargs):
        super().__init__()
        if norm_policy not in ('ema_bn', 'batch_stat', 'layer_norm'):
            raise ValueError('norm_policy must be ema_bn, batch_stat or layer_norm')
        self.norm_policy = norm_policy
        if ibot_reduction not in ('sample', 'token'):
            raise ValueError('Unknown iBOT reduction')
        self.ibot_reduction = ibot_reduction
        if patch_teacher_aggregation not in ('single_view',
                                             'matched_cross_view_mean'):
            raise ValueError('Unknown patch teacher aggregation')
        self.patch_teacher_aggregation = patch_teacher_aggregation
        if target_policy not in ('sinkhorn', 'ema_center'):
            raise ValueError('Unknown teacher target policy')
        self.target_policy = target_policy
        self.center_momentum = center_momentum
        if ibot_diversity_weight < 0:
            raise ValueError('iBOT diversity weight must be nonnegative')
        self.ibot_diversity_weight = ibot_diversity_weight
        if ibot_diversity_source not in ('masked', 'visible',
                                        'visible_position'):
            raise ValueError('Unknown iBOT diversity source')
        self.ibot_diversity_source = ibot_diversity_source
        if global_cls_policy not in ('masked', 'unmasked_separate'):
            raise ValueError('Unknown global CLS observation policy')
        if ibot_residual_gradient_policy not in ('inherit', 'full',
                                                 'bounded_branch'):
            raise ValueError('Unknown iBOT residual gradient policy')
        if (ibot_residual_gradient_policy != 'inherit' and
                global_cls_policy != 'unmasked_separate'):
            raise ValueError('Separate iBOT gradient policy requires separate CLS pass')
        self.global_cls_policy = global_cls_policy
        self.ibot_residual_gradient_policy = ibot_residual_gradient_policy
        if encoder_kwargs.get('block_variant', 'standard') == 'evidence':
            if norm_policy != 'layer_norm':
                raise ValueError('Evidence prototype requires token-local layer_norm')
            if global_cls_policy != 'masked' or ibot_residual_gradient_policy != 'inherit':
                raise ValueError('Evidence prototype currently uses the shared masked global pass')
        self.student = SpikingDinoEncoder(**encoder_kwargs)
        if norm_policy == 'batch_stat':
            self.student.apply(_use_current_batch_stats)
        elif norm_policy == 'layer_norm':
            _replace_bn_with_layer_norm(self.student)
        self.teacher = deepcopy(self.student).requires_grad_(False)
        if target_policy == 'ema_center':
            self.register_buffer('dino_center', torch.zeros(
                1, self.teacher.dino_head.prototypes.shape[0]))
            self.register_buffer('ibot_center', torch.zeros(
                1, self.teacher.ibot_head.prototypes.shape[0]))
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.teacher_momentum = teacher_momentum
        self.koleo_weight = koleo_weight
        self.gram_weight = gram_weight
        if balance_mode not in ('joint', 'per_view'):
            raise ValueError('balance_mode must be joint or per_view')
        self.balance_mode = balance_mode
        self.gram_teacher = None

    def train(self, mode=True):
        super().train(mode)
        if self.norm_policy in ('batch_stat', 'layer_norm'):
            self.teacher.eval()
        if self.gram_teacher is not None:
            self.gram_teacher.eval()
        return self

    @torch.no_grad()
    def enable_gram_refinement(self):
        """Snapshot the current EMA SNN teacher after its early training phase."""
        self.gram_teacher = deepcopy(self.teacher).requires_grad_(False)
        self.gram_teacher.eval()

    @torch.no_grad()
    def refresh_gram_teacher(self):
        if self.gram_teacher is None:
            raise RuntimeError('Gram refinement has not been enabled')
        self.gram_teacher.load_state_dict(self.teacher.state_dict())

    def forward(self, views, masks, return_components=False, diagnostics=False,
                context_ablation=False, temporal_masks=None):
        if len(views) < 2 or len(masks) != 2:
            raise ValueError('Need two global views and two matching masks')
        if views[0].shape != views[1].shape:
            raise ValueError('Global views must share a patch grid')
        if temporal_masks is not None:
            if len(temporal_masks) != 2:
                raise ValueError('Need two temporal mask tensors')
            for index, time_mask in enumerate(temporal_masks):
                if (time_mask.dtype != torch.bool or
                        time_mask.shape != (views[index].shape[0], *masks[index].shape) or
                        not torch.equal(time_mask.any(0), masks[index])):
                    raise ValueError('Temporal masks must select the same spatial patches')
        with torch.no_grad():
            union_mask = (masks[0] | masks[1]
                          if self.patch_teacher_aggregation ==
                          'matched_cross_view_mean' else None)
            teacher = [_independent(
                self.teacher, view,
                patch_head_mask=(union_mask if union_mask is not None
                                 else masks[i]))
                       for i, view in enumerate(views[:2])]
            teacher_cls_logits = torch.cat([v['cls_scores'] for v in teacher])
            if self.balance_mode == 'joint':
                cls_distribution = (sinkhorn_knopp(
                    teacher_cls_logits, self.teacher_temp)
                    if self.target_policy == 'sinkhorn' else
                    F.softmax((teacher_cls_logits - self.dino_center) /
                              self.teacher_temp, dim=-1))
                cls_target = list(cls_distribution.chunk(2))
            else:
                cls_target = [
                    (sinkhorn_knopp(v['cls_scores'], self.teacher_temp)
                     if self.target_policy == 'sinkhorn' else
                     F.softmax((v['cls_scores'] - self.dino_center) /
                               self.teacher_temp, dim=-1))
                    for v in teacher]
            if self.target_policy == 'ema_center':
                self.dino_center.lerp_(teacher_cls_logits.mean(0, keepdim=True),
                                       1 - self.center_momentum)
        if self.global_cls_policy == 'unmasked_separate':
            student = [_independent(self.student, v, compute_patch_head=False)
                       for v in views]
            blocks = self.student.backbone.blocks
            saved_policies = [block.residual_gradient_policy for block in blocks]
            if self.ibot_residual_gradient_policy != 'inherit':
                for block in blocks:
                    block.residual_gradient_policy = self.ibot_residual_gradient_policy
            try:
                patch_student = [_independent(
                    self.student, views[i],
                    (temporal_masks[i] if temporal_masks is not None
                     else masks[i]), patch_head_mask=masks[i],
                    patch_diversity=self.ibot_diversity_weight > 0,
                    patch_diversity_source=self.ibot_diversity_source,
                    compute_cls_head=False) for i in range(2)]
            finally:
                for block, policy in zip(blocks, saved_policies):
                    block.residual_gradient_policy = policy
        else:
            student = [_independent(self.student, v,
                                    (temporal_masks[i] if temporal_masks is not None
                                     else masks[i]) if i < 2 else None,
                                    patch_head_mask=masks[i] if i < 2 else None,
                                    compute_patch_head=i < 2,
                                    patch_diversity=i < 2 and
                                    self.ibot_diversity_weight > 0,
                                    patch_diversity_source=self.ibot_diversity_source)
                       for i, v in enumerate(views)]
            patch_student = student
        dino_terms = [-(target * F.log_softmax(s['cls_scores'] / self.student_temp,
                                              dim=-1)).sum(-1).mean()
                      for i, target in enumerate(cls_target)
                      for j, s in enumerate(student) if i != j]
        dino = torch.stack(dino_terms).mean()
        ibot_terms = []
        patch_logits = []
        masked_students = []
        for i in range(2):
            mask = masks[i]
            if mask.shape != teacher[i]['patch_spikes'].shape[1:3] or mask.dtype != torch.bool:
                raise ValueError('Mask and teacher patch grid do not align')
            if mask.any():
                if union_mask is None:
                    patch_logits.append(teacher[i]['patch_scores'])
                else:
                    selected = mask[union_mask]
                    patch_logits.append((
                        teacher[0]['patch_scores'][selected] +
                        teacher[1]['patch_scores'][selected]) * .5)
                masked_students.append(patch_student[i]['patch_scores'])
        if not patch_logits:
            raise ValueError('At least one patch must be masked')
        teacher_patch_logits = torch.cat(patch_logits)
        if self.target_policy == 'ema_center':
            patch_target = F.softmax((teacher_patch_logits - self.ibot_center) /
                                     self.teacher_temp, dim=-1)
            with torch.no_grad():
                self.ibot_center.lerp_(teacher_patch_logits.mean(0, keepdim=True),
                                       1 - self.center_momentum)
        else:
            patch_target = None
        if self.ibot_reduction == 'sample':
            target = (patch_target if patch_target is not None else
                      sinkhorn_knopp(teacher_patch_logits, self.teacher_temp)
                      if self.balance_mode == 'joint' else torch.cat([
                          sinkhorn_knopp(z, self.teacher_temp) for z in patch_logits]))
            ibot = masked_patch_cross_entropy(torch.cat(masked_students), target,
                                              masks, self.student_temp)
        elif self.balance_mode == 'joint':
            target = (patch_target if patch_target is not None else
                      sinkhorn_knopp(teacher_patch_logits, self.teacher_temp))
            prediction = F.log_softmax(torch.cat(masked_students) / self.student_temp,
                                       dim=-1)
            ibot = -(target * prediction).sum(-1).mean()
        else:
            centered_chunks = (patch_target.split([len(z) for z in patch_logits])
                               if patch_target is not None else None)
            target_chunks = []
            for index, (target_logits, student_logits) in enumerate(
                    zip(patch_logits, masked_students)):
                target = (centered_chunks[index] if centered_chunks is not None else
                          sinkhorn_knopp(target_logits, self.teacher_temp))
                target_chunks.append(target)
                prediction = F.log_softmax(student_logits / self.student_temp, dim=-1)
                ibot_terms.append(-(target * prediction).sum(-1).mean())
            ibot = torch.stack(ibot_terms).mean()
            target = torch.cat(target_chunks)
        crop_cls_rates = [s['cls_spikes'].float().mean(0) for s in student[:2]]
        cls_rates = torch.cat(crop_cls_rates)
        # DINOv3 applies KoLeo inside each global crop. Concatenating both
        # views would repel two augmentations of the very same recording.
        koleo = torch.stack([koleo_loss(rate) for rate in crop_cls_rates]).sum()
        total = dino + ibot + self.koleo_weight * koleo
        result = {'dino': dino.detach(), 'ibot': ibot.detach(),
                  'koleo': koleo.detach()}
        if self.ibot_diversity_weight > 0:
            diversity_terms = [s['patch_diversity_gap'] for s in patch_student[:2]
                               if s['patch_diversity_gap'] is not None]
            diversity = torch.stack(diversity_terms).mean()
            total = total + self.ibot_diversity_weight * diversity
            result['ibot_diversity_gap'] = diversity.detach()
        with torch.no_grad():
            targets = torch.cat(cls_target)
            sample_entropy = -(targets * targets.clamp_min(1e-12).log()).sum(-1).mean()
            marginal = targets.mean(0)
            marginal_entropy = -(marginal * marginal.clamp_min(1e-12).log()).sum()
            result['teacher_target_entropy'] = sample_entropy
            result['teacher_marginal_entropy'] = marginal_entropy
            result['teacher_target_information'] = marginal_entropy - sample_entropy
            result['teacher_top1_mass'] = targets.max(-1).values.mean()
            result['cls_spike_rate'] = student[0]['cls_spikes'].float().mean()
            result['patch_spike_rate'] = patch_student[0]['patch_spikes'].float().mean()
            result['cls_feature_std'] = cls_rates.std(0, unbiased=False).mean()
            patch_marginal = target.mean(0)
            patch_entropy = -(target * target.clamp_min(1e-12).log()).sum(-1).mean()
            patch_marginal_entropy = -(
                patch_marginal * patch_marginal.clamp_min(1e-12).log()).sum()
            result['teacher_patch_target_information'] = (
                patch_marginal_entropy - patch_entropy)
            if diagnostics:
                masked_predictions = torch.cat(masked_students)
                matched_patch_loss = masked_patch_cross_entropy(
                    masked_predictions, target, masks, self.student_temp)
                shuffled_target = target.roll(max(1, len(target) // 2), dims=0)
                shuffled_patch_loss = masked_patch_cross_entropy(
                    masked_predictions, shuffled_target, masks,
                    self.student_temp)
                result['ibot_matched_patch_ce'] = matched_patch_loss
                result['ibot_shuffled_patch_ce'] = shuffled_patch_loss
                result['ibot_context_advantage'] = (shuffled_patch_loss -
                                                     matched_patch_loss)
                mask_coordinates = torch.cat(masks).nonzero(as_tuple=False)
                positions = mask_coordinates[:, 1]
                position_count = masks[0].shape[1]
                counts = torch.bincount(positions, minlength=position_count)
                sums = target.new_zeros(position_count, target.shape[-1])
                sums.index_add_(0, positions, target)
                means = sums / counts.clamp_min(1)[:, None]
                mean_entropy = -(
                    means * means.clamp_min(1e-12).log()).sum(-1)
                row_entropy = -(
                    target * target.clamp_min(1e-12).log()).sum(-1)
                within_position = (
                    mean_entropy[positions] - row_entropy).mean()
                result['teacher_patch_within_position_information'] = (
                    within_position)
                result['teacher_patch_position_information'] = (
                    result['teacher_patch_target_information'] - within_position)
                position_groups = {}
                for token_index, (sample_crop, position) in enumerate(
                        mask_coordinates.tolist()):
                    position_groups.setdefault(position, []).append(
                        (token_index, sample_crop % masks[0].shape[0]))
                eligible = []
                wrong = []
                for group in position_groups.values():
                    for token_index, recording in group:
                        alternative = next((other for other, other_recording in group
                                            if other_recording != recording), None)
                        if alternative is not None:
                            eligible.append(token_index)
                            wrong.append(alternative)
                if eligible:
                    index = torch.tensor(eligible, device=target.device)
                    other = torch.tensor(wrong, device=target.device)
                    prediction = F.log_softmax(
                        masked_predictions.float() / self.student_temp, dim=-1)
                    matched_ce = -(target[index] * prediction[index]).sum(-1)
                    wrong_ce = -(target[other] * prediction[index]).sum(-1)
                    result['ibot_same_position_advantage'] = (
                        wrong_ce - matched_ce).mean()
                    result['ibot_same_position_pairs'] = target.new_tensor(
                        len(eligible))
                    if context_ablation:
                        zero_outputs = [_independent(
                            self.student, torch.zeros_like(views[i]),
                            mask=(temporal_masks[i] if temporal_masks is not None
                                  else masks[i]), patch_head_mask=masks[i],
                            compute_patch_head=True, patch_diversity=False)
                            for i in range(2)]
                        zero_scores = torch.cat([
                            v['patch_scores'] for v in zero_outputs])
                        result['ibot_zero_event_logit_change'] = (
                            zero_scores.float() - masked_predictions.float()
                        ).abs().mean()
                        result['ibot_zero_event_logit_change_relative'] = (
                            result['ibot_zero_event_logit_change'] /
                            masked_predictions.float().std(
                                0, unbiased=False).mean().clamp_min(1e-6))
                        zero_prediction = F.log_softmax(
                            zero_scores.float() / self.student_temp, dim=-1)
                        zero_matched = -(
                            target[index] * zero_prediction[index]).sum(-1)
                        zero_wrong = -(
                            target[other] * zero_prediction[index]).sum(-1)
                        result['ibot_zero_event_same_position_advantage'] = (
                            zero_wrong - zero_matched).mean()
                raw_patch_target = F.softmax(teacher_patch_logits.float() /
                                             self.teacher_temp, dim=-1)
                raw_patch_marginal = raw_patch_target.mean(0)
                result['teacher_patch_softmax_information'] = -(
                    raw_patch_marginal * raw_patch_marginal.clamp_min(1e-12).log()
                ).sum() + (
                    raw_patch_target * raw_patch_target.clamp_min(1e-12).log()
                ).sum(-1).mean()
                result['teacher_patch_logit_std'] = teacher_patch_logits.float().std(
                    0, unbiased=False).mean()
                selected_teacher_patch = torch.cat([
                    teacher[i]['patch_spikes'][:, masks[i]] for i in range(2)
                    if masks[i].any()], dim=1).float().mean(0)
                result['teacher_patch_feature_std'] = selected_teacher_patch.std(
                    0, unbiased=False).mean()
            masked_scores = torch.cat(masked_students)
            result['masked_student_logit_std'] = masked_scores.float().std(
                0, unbiased=False).mean()
            masked_patch_spikes = torch.cat([
                patch_student[i]['patch_spikes'][:, masks[i]] for i in range(2)
                if masks[i].any()], dim=1)
            result['masked_patch_spike_rate'] = masked_patch_spikes.float().mean()
        if self.gram_teacher is not None:
            # The snapshot's BN buffers are part of the frozen target too.
            self.gram_teacher.eval()
            gram_terms = []
            for view in views[:2]:
                with torch.no_grad():
                    anchor = _independent(self.gram_teacher, view,
                                          compute_patch_head=False)['patch_spikes'].float().mean(0)
                # Use an unmasked student pass: a masked patch cannot be
                # expected to preserve its observed local geometry.
                patch = _independent(self.student, view,
                                     compute_patch_head=False)['patch_spikes'].float().mean(0)
                gram_terms.append(gram_loss(patch, anchor))
            gram = torch.stack(gram_terms).mean()
            total = total + self.gram_weight * gram
            result['gram'] = gram.detach()
            if return_components:
                result['_gram_raw'] = gram
        if return_components:
            # Diagnostic-only references to the original autograd graph.
            result['_dino_raw'] = dino
            result['_ibot_raw'] = ibot
            result['_koleo_raw'] = koleo
            if self.ibot_diversity_weight > 0:
                result['_diversity_raw'] = diversity
        result['loss'] = total
        return result

    @torch.no_grad()
    def after_optimizer_step(self):
        for target, source in zip(self.teacher.parameters(), self.student.parameters()):
            target.lerp_(source, 1 - self.teacher_momentum)
        for target, source in zip(self.teacher.buffers(), self.student.buffers()):
            target.copy_(source)
