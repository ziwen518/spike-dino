"""Spike-DINO v4: all-spiking ViT student/teacher with a DINO recipe.

Backbone is the validated SPS+BN 12x384/6-head Spikformer-count ViT with
LayerScale .1 (supervised dev 49.15%). Every tensor crossing between synaptic
layers is a binary spike; BN currents, real membrane residuals, integer Q(KᵀV)
counts and the continuous prototype/centering loss readout are disclosed.

Differences from E2 (spike_dino/e2_dino.py, kept as negative evidence):
* head input = binary CLS spikes ++ BN+LIF re-encoded mean patch current
  (the saturating 196-patch OR route is removed);
* DINO-official BN behaviour: student and teacher both normalize with the
  current batch during training, each view in its own forward pass, so global
  and local statistics never mix; evaluation always uses precise-BN
  calibration on a fixed, label-free, class-mixed order;
* externally scheduled teacher temperature and EMA momentum;
* frozen downstream representation = CLS spike rate ++ GAP patch spike rate.
"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.activation_based import functional

from .dinov3 import SpikingProjectionHead, TimeBatchNorm
from .fused_lif import replace_lif
from .spiking_vit import SpikingVisionTransformer, _lif


def _sew_backbone(**kwargs):
    # Imported lazily: the ViT line must not depend on spikingjelly's model zoo.
    from .sdino_sew import SEWResNetBackbone
    return SEWResNetBackbone(**kwargs)


class SpikeDinoV4Encoder(nn.Module):
    def __init__(self, *, image_size=224, patch_size=16, dim=384, depth=12,
                 heads=6, prototypes=4096, head_hidden=2048, bottleneck=256,
                 fused_lif=True, rate_centering=False, rate_center_momentum=.9,
                 embedding_source='head', head_input='spikes', backbone='vit'):
        super().__init__()
        # head_input='current': the spiking projection head receives the BN
        # currents entering the backbone's final CLS/patch LIFs (time series,
        # patch current averaged over tokens) instead of their binary spikes.
        if head_input not in ('spikes', 'current'):
            raise ValueError('Unknown head input')
        if backbone not in ('vit', 'sew_resnet18'):
            raise ValueError('Unknown backbone')
        self.head_input = head_input
        self.backbone_kind = backbone
        if backbone == 'sew_resnet18':
            # Fixed by the SEW-ResNet-18 width; no class token, so the spiking
            # head sees layer-4 GAP ++ LIF(BN(layer-4 GAP)) instead of CLS ++
            # BN+LIF(patch mean). FusedLIF is ViT-only (SEW neurons are
            # stateful [T*N,C,H,W] nodes), and the 'current' routes are absent.
            dim = 512
            fused_lif = False
            if head_input != 'spikes':
                raise ValueError('SEW-ResNet-18 supports the spiking head input only')
            if embedding_source == 'current':
                raise ValueError('SEW-ResNet-18 has no current embedding')
        self.dim = dim
        # Contrastive embedding: 'head' = centered bottleneck rate (as DINO);
        # 'backbone' = centered CLS ++ GAP spike rate (the frozen-probe feature).
        if embedding_source not in ('head', 'backbone', 'current'):
            raise ValueError('Unknown embedding source')
        self.embedding_source = embedding_source
        if embedding_source in ('backbone', 'current'):
            self.register_buffer('rep_center', torch.zeros(2 * dim))
        self.backbone = SpikingVisionTransformer(
            image_size=image_size, patch_size=patch_size, in_channels=2,
            dim=dim, depth=depth, heads=heads, mlp_ratio=4,
            attention_variant='spikformer_count',
            cls_aggregation='attention_only', residual_policy='layer_scale',
            residual_gradient_policy='full', tokenizer='sps',
            layer_scale_init=.1) if backbone == 'vit' else _sew_backbone()
        self.patch_pool_norm = TimeBatchNorm(dim)
        self.patch_pool_spike = _lif(threshold=.25)
        self.head = SpikingProjectionHead(
            dim * 2, hidden=head_hidden, bottleneck=bottleneck,
            prototypes=prototypes, layers=3, prototype_readout='cosine')
        # v4d: the non-negative bottleneck spike rate r is centered by a
        # running mean (an EMA buffer updated only from Student global crops
        # and copied to the Teacher) before L2 normalization, so the cosine
        # prototype readout can use signed directions. The spiking head is
        # unchanged; with rate_centering=False the readout is exactly v4b.
        self.rate_centering = bool(rate_centering)
        self.rate_center_momentum = float(rate_center_momentum)
        if self.rate_centering:  # absent otherwise, so v4/v4b/v4c checkpoints load unchanged
            self.register_buffer('rate_center', torch.zeros(bottleneck))
        self.fused_lif_modules = replace_lif(self) if fused_lif else 0
        # 'current': contrastive readout = time-mean BN current entering the
        # backbone's final CLS/patch LIFs (real-valued loss readout; all
        # inter-layer communication stays binary).
        self.backbone.keep_output_current = (embedding_source == 'current'
                                             or head_input == 'current')

    def encode(self, events):
        cls, patch = self.backbone(events)
        pooled = self.patch_pool_spike(self.patch_pool_norm(patch.mean(2)))
        return cls, patch, pooled

    def head_scores(self, spikes, update_rate_center=False, return_feature=False):
        head = self.head
        h = head.lif1(head.bn1(head.fc1(spikes)))
        if head.fc_mid is not None:
            h = head.lif_mid(head.bn_mid(head.fc_mid(h)))
        h = head.lif2(head.bn2(head.fc2(h)))
        rate = h.float().mean(0)
        readout = rate
        if self.rate_centering:
            readout = rate - self.rate_center
            if update_rate_center:
                with torch.no_grad():
                    self.rate_center.lerp_(rate.detach().mean(0),
                                           1 - self.rate_center_momentum)
        feature = F.normalize(readout, dim=-1, eps=1e-6)
        weight = F.normalize(head.prototypes, dim=-1, eps=1e-6)
        if return_feature:
            return F.linear(feature, weight), rate, feature
        return F.linear(feature, weight), rate

    def forward(self, events, update_rate_center=False):
        cls, patch, pooled = self.encode(events)
        currents = None
        if self.backbone.keep_output_current:
            currents = self.backbone.output_current
            self.backbone.output_current = None
        if self.head_input == 'current':
            spikes = torch.cat((currents[0], currents[1].mean(2)), dim=-1)
        else:
            spikes = torch.cat((cls, pooled), dim=-1)
        scores, rate, feature = self.head_scores(spikes, update_rate_center,
                                                 return_feature=True)
        if self.embedding_source in ('backbone', 'current'):
            if self.embedding_source == 'current':
                cls_current, patch_current = currents
                rep = torch.cat((cls_current.float().mean(0),
                                 patch_current.float().mean((0, 2))), -1)
            else:
                rep = torch.cat((cls.float().mean(0), patch.float().mean((0, 2))), -1)
            feature = F.normalize(rep - self.rep_center, dim=-1, eps=1e-6)
            if update_rate_center:
                with torch.no_grad():
                    self.rep_center.lerp_(rep.detach().mean(0), 1 - self.rate_center_momentum)
        return {'scores': scores, 'cls_spikes': cls,
                'patch_spikes': patch, 'pooled_spikes': pooled,
                'head_input_spikes': spikes, 'bottleneck_rate': rate,
                'embedding': feature}

    @property
    def classifier_dim(self):
        """Width of the supervised fine-tuning input (gradients flow through)."""
        if self.backbone_kind == 'sew_resnet18':
            return self.dim + self.backbone.mid_dim
        return 2 * self.dim

    def classifier_features(self, events):
        if self.backbone_kind == 'sew_resnet18':
            from .sdino_sew import sew_classifier_features
            return sew_classifier_features(self, events)
        cls, patch, _ = self.encode(events)
        return torch.cat((cls.float().mean(0), patch.float().mean((0, 2))), -1)

    @torch.no_grad()
    def representation(self, events):
        """Frozen downstream features: spike rates (and final-LIF input
        currents as 'concat_current'), no head."""
        if self.backbone_kind == 'sew_resnet18':
            from .sdino_sew import sew_representation
            return sew_representation(self, events)
        keep = self.backbone.keep_output_current
        self.backbone.keep_output_current = True
        try:
            cls, patch, _ = self.encode(events)
            cls_current, patch_current = self.backbone.output_current
        finally:
            self.backbone.output_current = None
            self.backbone.keep_output_current = keep
        cls_rate = cls.float().mean(0)
        patch_rate = patch.float().mean((0, 2))
        return {'cls': cls_rate, 'gap': patch_rate,
                'concat': torch.cat((cls_rate, patch_rate), -1),
                'concat_current': torch.cat((cls_current.float().mean(0),
                                             patch_current.float().mean((0, 2))), -1)}


class _running_stats_frozen:
    """Temporarily set BN momentum to 0: batch-stat normalization, no buffer update."""

    def __init__(self, module, active):
        self.norms = ([m for m in module.modules()
                       if isinstance(m, nn.modules.batchnorm._BatchNorm)]
                      if active else [])
        self.saved = []

    def __enter__(self):
        self.saved = [(m, m.momentum, m.num_batches_tracked.clone()
                       if m.num_batches_tracked is not None else None)
                      for m in self.norms]
        for module in self.norms:
            if module.momentum is None:
                raise RuntimeError('Cumulative BN momentum cannot be frozen')
            module.momentum = 0.
        return self

    def __exit__(self, *exc):
        for module, momentum, tracked in self.saved:
            module.momentum = momentum
            if tracked is not None:
                module.num_batches_tracked.copy_(tracked)
        return False


def independent_forward(model, events, **kwargs):
    functional.reset_net(model)
    try:
        return model(events, **kwargs)
    finally:
        functional.reset_net(model)


class SpikeDinoV4(nn.Module):
    def __init__(self, *, student_temp=.1, center_momentum=.9,
                 teacher_bn_mode='train', koleo_weight=0.,
                 bottleneck_var_weight=0., bottleneck_var_gamma=.15,
                 bottleneck_decorr_weight=0., objective='dino',
                 moco_queue=8192, moco_temp=.2, moco_weight=1., analog='none',
                 **encoder_kwargs):
        super().__init__()
        if teacher_bn_mode not in ('train', 'running_eval'):
            raise ValueError('Unknown Teacher BN mode')
        if koleo_weight < 0:
            raise ValueError('KoLeo weight must be nonnegative')
        self.teacher_bn_mode = teacher_bn_mode
        self.koleo_weight = float(koleo_weight)
        if bottleneck_var_weight < 0 or bottleneck_decorr_weight < 0:
            raise ValueError('Bottleneck regularizer weights must be nonnegative')
        self.bottleneck_var_weight = float(bottleneck_var_weight)
        self.bottleneck_var_gamma = float(bottleneck_var_gamma)
        self.bottleneck_decorr_weight = float(bottleneck_decorr_weight)
        # analog != 'none': parameter-matched ANN control (every LIF -> GELU;
        # 'softmax' also swaps count attention for softmax attention). This is
        # a diagnostic counterfactual only, never the Spike-DINO model.
        if analog not in ('none', 'count', 'softmax', 'spiking_softmax_attention'):
            raise ValueError('Unknown analog control')
        self.analog = analog
        if analog != 'none' and encoder_kwargs.get('backbone', 'vit') != 'vit':
            raise ValueError('Analog controls are defined for the ViT backbone only')
        if analog == 'none':
            self.student = SpikeDinoV4Encoder(**encoder_kwargs)
        elif analog == 'spiking_softmax_attention':
            from .analog_control import make_analog_control
            self.student = make_analog_control(
                SpikeDinoV4Encoder(**encoder_kwargs), 'softmax', replace_neurons=False)
        else:
            from .analog_control import make_analog_control
            self.student = make_analog_control(
                SpikeDinoV4Encoder(**{**encoder_kwargs, 'fused_lif': False}), analog)
        self.teacher = deepcopy(self.student).requires_grad_(False)
        prototypes = self.student.head.prototypes.shape[0]
        self.register_buffer('center', torch.zeros(1, prototypes))
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        # Control objective: MoCo-v2-style InfoNCE on the same all-spiking
        # Student/EMA-Teacher, views and spiking head. Embedding = normalized
        # (centered) bottleneck spike rate; negatives = a queue of Teacher keys.
        # 'dino_moco': Spike-DINO distillation plus moco_weight x InfoNCE on the
        # same embedding (DINO remains the primary objective).
        if objective not in ('dino', 'moco', 'dino_moco'):
            raise ValueError('Unknown objective')
        self.objective = objective
        self.moco_temp = float(moco_temp)
        self.moco_weight = float(moco_weight)
        if objective in ('moco', 'dino_moco'):
            dim = (self.student.head.fc2.out_features
                   if self.student.embedding_source == 'head' else 2 * self.student.dim)  # backbone/current
            self.register_buffer('moco_queue', F.normalize(
                torch.randn(int(moco_queue), dim), dim=-1))
            self.register_buffer('moco_pointer', torch.zeros((), dtype=torch.long))

    def train(self, mode=True):
        # DINO-official: the teacher also normalizes with batch statistics
        # during training; it never receives gradients.
        super().train(mode)
        if self.teacher_bn_mode == 'running_eval':
            self.teacher.eval()
        return self

    def forward(self, views, teacher_temp, autocast_dtype=None):
        if len(views) < 2 or views[0].shape != views[1].shape:
            raise ValueError('Need two same-size globals followed by locals')
        autocast = (torch.autocast('cuda', dtype=autocast_dtype)
                    if autocast_dtype is not None else torch.autocast(
                        'cuda', enabled=False))
        batch = views[0].shape[1]
        with autocast:
            # DINO multi-crop: same-resolution crops share one forward pass
            # (batch dim 1 of [T,B,C,H,W]); global and local BN statistics stay
            # separate because they are never in the same pass.
            with torch.no_grad():
                teacher_out = independent_forward(self.teacher, torch.cat(views[:2], 1))
                teacher_logits = teacher_out['scores'].float()
                teacher_rate = teacher_out['bottleneck_rate'].float()
                teacher_embedding = teacher_out['embedding'].float()
            groups = [views[:2]] + ([views[2:]] if len(views) > 2 else [])
            student = []
            for group_index, group in enumerate(groups):
                # With an eval-mode Teacher, the Student running statistics
                # are copied into the Teacher every step. Only the global
                # pass may update them (local 96px crops still normalize with
                # their own batch statistics, but leave running buffers
                # untouched), so Teacher global targets never use local stats.
                freeze_running = (self.teacher_bn_mode == 'running_eval'
                                  and group_index > 0)
                with _running_stats_frozen(self.student, freeze_running):
                    # The rate center, like BN running stats, is updated only
                    # from Student global crops (after they used the old one).
                    output = independent_forward(
                        self.student, torch.cat(group, 1),
                        update_rate_center=self.training and group_index == 0)
                if group_index == 0:
                    student_global_rate = output['bottleneck_rate'].float()
                for index in range(len(group)):
                    part = slice(index * batch, (index + 1) * batch)
                    student.append({
                        'scores': output['scores'][part],
                        'cls_spikes': output['cls_spikes'][:, part],
                        'patch_spikes': output['patch_spikes'][:, part],
                        'pooled_spikes': output['pooled_spikes'][:, part],
                        'embedding': output['embedding'][part]})
        targets_all = F.softmax((teacher_logits - self.center) / teacher_temp, -1)
        targets = targets_all.chunk(2)
        terms, global_terms, local_terms = [], [], []
        for teacher_index, target in enumerate(targets):
            for student_index, output in enumerate(student):
                if student_index == teacher_index:
                    continue
                term = -(target * F.log_softmax(
                    output['scores'].float() / self.student_temp, -1)).sum(-1).mean()
                terms.append(term)
                (global_terms if student_index < 2 else local_terms).append(term)
        dino = torch.stack(terms).mean()
        # KoLeo (DINOv2) on the frozen-probe representation itself: the
        # L2-normalized CLS spike rate ++ GAP patch spike rate of each Student
        # global crop, applied per crop. Gradients reach the backbone through
        # the LIF surrogate. Weight 0 reproduces v4b exactly.
        koleo = dino.new_zeros(())
        if self.koleo_weight > 0:
            koleo = torch.stack([koleo_loss(torch.cat((
                output['cls_spikes'].float().mean(0),
                output['patch_spikes'].float().mean((0, 2))), -1))
                for output in student[:2]]).mean()
        # v4e: anti-collapse on the Student 256-d bottleneck spike rate of the
        # two global crops (2B samples): a per-dimension std floor (hinge at
        # gamma) plus the mean squared off-diagonal correlation. The std floor
        # is what rules out the constant solution; decorrelation alone cannot.
        # Both are invariant to the rate center. Weights 0 reproduce v4d.
        bottleneck_var, bottleneck_decorr = bottleneck_regularizers(
            student_global_rate, self.bottleneck_var_gamma)
        moco = dino.new_zeros(())
        moco_accuracy = dino.new_zeros(())
        if self.objective in ('moco', 'dino_moco'):
            keys = teacher_embedding.chunk(2)
            queue = self.moco_queue.clone().detach()
            moco_terms, hits = [], []
            for student_index, output in enumerate(student):
                query = output['embedding'].float()
                for key_index, key in enumerate(keys):
                    if student_index == key_index:
                        continue
                    positive = (query * key).sum(-1, keepdim=True)
                    logits = torch.cat((positive, query @ queue.T), 1) / self.moco_temp
                    moco_terms.append(F.cross_entropy(
                        logits, logits.new_zeros(len(logits), dtype=torch.long)))
                    hits.append((logits.argmax(-1) == 0).float().mean().detach())
            moco = torch.stack(moco_terms).mean()
            moco_accuracy = torch.stack(hits).mean()
            if self.training:
                with torch.no_grad():
                    new = teacher_embedding.detach()
                    size = len(self.moco_queue)
                    index = (self.moco_pointer + torch.arange(len(new), device=new.device)) % size
                    self.moco_queue[index] = new.to(self.moco_queue.dtype)
                    self.moco_pointer.copy_((self.moco_pointer + len(new)) % size)
        main_loss = {'dino': dino, 'moco': moco,
                     'dino_moco': dino + self.moco_weight * moco}[self.objective]
        loss = (main_loss + self.koleo_weight * koleo
                + self.bottleneck_var_weight * bottleneck_var
                + self.bottleneck_decorr_weight * bottleneck_decorr)
        with torch.no_grad():
            self.center.lerp_(teacher_logits.mean(0, keepdim=True),
                              1 - self.center_momentum)
            marginal = targets_all.mean(0)
            entropy = -(targets_all * targets_all.clamp_min(1e-12).log()).sum(-1).mean()
            marginal_entropy = -(marginal * marginal.clamp_min(1e-12).log()).sum()
            student_probs = F.softmax(student[0]['scores'].float() / self.student_temp, -1)
            cls_rate = student[0]['cls_spikes'].float().mean(0)
            metrics = {
                'loss': loss.detach(),
                'dino': dino.detach(),
                'koleo': koleo.detach(),
                'moco': moco.detach(),
                'moco_accuracy': moco_accuracy,
                'positive_cosine': (student[0]['embedding'].float()
                                    * teacher_embedding.chunk(2)[1]).sum(-1).mean(),
                'bottleneck_var_loss': bottleneck_var.detach(),
                'bottleneck_decorr': bottleneck_decorr.detach(),
                'dino_global': torch.stack(global_terms).mean(),
                'dino_local': (torch.stack(local_terms).mean() if local_terms
                               else loss.new_zeros(())),
                'teacher_entropy': entropy,
                'teacher_marginal_entropy': marginal_entropy,
                'teacher_information': marginal_entropy - entropy,
                'teacher_top1_mass': targets_all.max(-1).values.mean(),
                'student_entropy': -(student_probs * student_probs.clamp_min(
                    1e-12).log()).sum(-1).mean(),
                'prototype_usage': targets_all.new_tensor(
                    targets_all.argmax(-1).unique().numel() / targets_all.shape[-1]),
                'cls_spike_rate': cls_rate.mean(),
                'patch_spike_rate': student[0]['patch_spikes'].float().mean(),
                'pooled_spike_rate': student[0]['pooled_spikes'].float().mean(),
                'cls_feature_std': cls_rate.std(0, unbiased=False).mean(),
                # v4d diagnostics: prototype-logit dynamic range of the Teacher
                # (before temperature) and spread of its 256-d bottleneck rate.
                'teacher_logit_std': (teacher_logits - self.center).std(-1).mean(),
                'teacher_bottleneck_rate': teacher_rate.mean(),
                'teacher_bottleneck_std': teacher_rate.std(0, unbiased=False).mean(),
            }
        return loss, metrics

    @torch.no_grad()
    def update_teacher(self, momentum):
        for target, source in zip(self.teacher.parameters(),
                                  self.student.parameters()):
            target.lerp_(source.detach(), 1 - momentum)
        for target, source in zip(self.teacher.buffers(), self.student.buffers()):
            target.copy_(source)


def model_from_config(config, **overrides):
    """Rebuild a SpikeDinoV4 from a checkpoint config (older runs lack newer keys)."""
    kwargs = dict(prototypes=config.get('prototypes', 4096),
                  image_size=config.get('image_size', 224),
                  teacher_bn_mode=config.get('teacher_bn_mode', 'train'),
                  koleo_weight=config.get('koleo_weight', 0.),
                  rate_centering=config.get('rate_centering', False),
                  rate_center_momentum=config.get('rate_center_momentum', .9),
                  bottleneck_var_weight=config.get('bottleneck_var_weight', 0.),
                  bottleneck_var_gamma=config.get('bottleneck_var_gamma', .15),
                  bottleneck_decorr_weight=config.get('bottleneck_decorr_weight', 0.),
                  objective=config.get('objective', 'dino'),
                  moco_queue=config.get('moco_queue', 8192),
                  moco_temp=config.get('moco_temp', .2),
                  moco_weight=config.get('moco_weight', 1.),
                  embedding_source=config.get('embedding_source', 'head'),
                  backbone=config.get('backbone_arch', 'vit'),
                  analog=config.get('analog', 'none'),
                  # older configs used 'head_input' as a free-text description
                  head_input=(config.get('head_input') if config.get('head_input')
                              in ('spikes', 'current') else 'spikes'))
    kwargs.update(overrides)
    return SpikeDinoV4(**kwargs)


def bottleneck_regularizers(rate, gamma, eps=1e-4):
    """(mean hinge(gamma - std_j), mean_{i!=j} corr_ij^2) over the batch dim."""
    centered = rate - rate.mean(0, keepdim=True)
    std = (centered.pow(2).mean(0) + eps).sqrt()
    variance = F.relu(gamma - std).mean()
    z = centered / std
    corr = z.T @ z / rate.shape[0]
    d = corr.shape[0]
    off = corr - torch.diag_embed(torch.diagonal(corr))
    return variance, off.pow(2).sum() / (d * (d - 1))


def koleo_loss(features, eps=1e-8):
    """Kozachenko-Leonenko entropy estimator as in DINOv2 (per batch)."""
    features = F.normalize(features, dim=-1, eps=eps)
    with torch.no_grad():
        similarity = features @ features.T
        similarity.fill_diagonal_(-2.)
        neighbour = similarity.argmax(1)
    distance = F.pairwise_distance(features, features[neighbour], p=2, eps=eps)
    return -torch.log(distance + eps).mean()


def cosine_schedule(base, final, total_steps, warmup_steps=0, warmup_start=0.):
    """Per-step values: linear warmup then cosine from base to final."""
    import math
    values = []
    for step in range(total_steps):
        if step < warmup_steps:
            values.append(warmup_start + (base - warmup_start) * step / warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            values.append(final + .5 * (base - final) * (1 + math.cos(math.pi * progress)))
    return values


def teacher_temperature(step, steps_per_epoch, warmup_epochs, start=.04, final=.07):
    warmup = warmup_epochs * steps_per_epoch
    if warmup <= 0 or step >= warmup:
        return final
    return start + (final - start) * step / warmup
