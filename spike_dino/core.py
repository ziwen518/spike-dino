"""Single-device research baseline; tensors use [T, B, C, H, W]."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.activation_based import functional, layer, neuron, surrogate

from third_party.eventclr.models.sew_resnet import SEWResNet18
from third_party.eventclr.models.pretraining import EventCLR
from third_party.eventclr.ntxent import TemporalNTXentLoss


class TemporalBatchNorm1d(layer.BatchNorm1d):
    """Support [T,B,C] on SpikingJelly .14, matching pooled T*B statistics."""
    def forward(self, x):
        if self.step_mode == 'm' and x.ndim == 3:
            return nn.BatchNorm1d.forward(self, x.flatten(0, 1)).reshape_as(x)
        return super().forward(x)


def adapt_projection_bn(model):
    for name, child in list(model.named_children()):
        if type(child) is layer.BatchNorm1d:
            replacement = TemporalBatchNorm1d(child.num_features, eps=child.eps,
                                             momentum=child.momentum, step_mode='m')
            replacement.load_state_dict(child.state_dict())
            setattr(model, name, replacement)
        else:
            adapt_projection_bn(child)


def backbone(width=1.0):
    return SEWResNet18(width=width, cnf='ADD', neuron_type='LIF',
                       tau=2.0, v_threshold=1.0, v_reset=0.0,
                       surrogate_function=surrogate.ATan(), detach_reset=True,
                       backend='torch')


class StrictSpikeBackbone(SEWResNet18):
    """SEW-ResNet18 with binary residual output and a spike-coded pooled feature."""
    def __init__(self, width=1.0):
        # The upstream SEW blocks initialize their second BN scale to zero.
        # IAND preserves the spiking identity path at initialization; AND
        # would multiply it by an all-zero residual and silence the network.
        super().__init__(width=width, cnf='IAND', neuron_type='LIF',
                         tau=2.0, v_threshold=1.0, v_reset=0.0,
                         surrogate_function=surrogate.ATan(), detach_reset=True,
                         backend='torch')
        # Global average pooling is a membrane current; re-encode before the head.
        self.readout_lif = neuron.LIFNode(
            tau=2.0, v_threshold=.05, v_reset=0.0, detach_reset=True,
            surrogate_function=surrogate.ATan(), backend='torch')
        functional.set_step_mode(self, 'm')

    def forward(self, x):
        return self.readout_lif(super().forward(x))


class DinoHead(nn.Module):
    """Spiking hidden layer, continuous bottleneck and normalized prototypes.

    This is a candidate head, not a reproduction of the original DINO MLP.
    """
    def __init__(self, features, hidden=2048, bottleneck=256, prototypes=1024):
        super().__init__()
        self.layers = nn.Sequential(
            layer.Linear(features, hidden), TemporalBatchNorm1d(hidden),
            neuron.LIFNode(tau=2.0, v_reset=0.0, detach_reset=True,
                           surrogate_function=surrogate.ATan(), backend='torch'),
            layer.Linear(hidden, bottleneck))
        self.prototypes = nn.Parameter(torch.randn(prototypes, bottleneck) * .02)
        functional.set_step_mode(self, 'm')

    def forward(self, x):
        x = F.normalize(self.layers(x), dim=-1, eps=1e-6)
        return F.linear(x, F.normalize(self.prototypes, dim=-1, eps=1e-6))


class StrictSpikeDinoHead(nn.Module):
    """All communicated head activations are binary spikes.

    The final prototype neurons emit spikes; DINO computes its target
    distributions from their firing rates outside the spiking forward path.
    """
    def __init__(self, features, hidden=2048, prototypes=1024, population=1):
        super().__init__()
        if population < 1:
            raise ValueError('Population size must be positive')
        self.population = population
        self.hidden_linear = layer.Linear(features, hidden)
        self.hidden_bn = TemporalBatchNorm1d(hidden)
        self.hidden_lif = neuron.LIFNode(
            tau=2.0, v_threshold=1.0, v_reset=0.0, detach_reset=True,
            surrogate_function=surrogate.ATan(), backend='torch')
        self.prototype_linear = layer.Linear(hidden, prototypes * population)
        self.prototype_bn = TemporalBatchNorm1d(prototypes * population)
        self.prototype_lif = neuron.LIFNode(
            tau=2.0, v_threshold=1.0, v_reset=0.0, detach_reset=True,
            surrogate_function=surrogate.ATan(), backend='torch')
        self.register_buffer('prototype_threshold', torch.tensor(1.0))
        self.last_rates = {}
        functional.set_step_mode(self, 'm')

    def forward(self, x):
        hidden = self.hidden_lif(self.hidden_bn(self.hidden_linear(x)))
        self.prototype_lif.v_threshold = float(self.prototype_threshold)
        spikes = self.prototype_lif(self.prototype_bn(self.prototype_linear(hidden)))
        self.last_rates = {
            'head_hidden_rate': hidden.detach().float().mean().item(),
            'head_prototype_rate': spikes.detach().float().mean().item(),
        }
        return spikes

    @torch.no_grad()
    def adjust_output_threshold(self, target_rate, gain=.25):
        rate = self.last_rates['head_prototype_rate']
        self.prototype_threshold.mul_(torch.exp(
            self.prototype_threshold.new_tensor(gain * (rate - target_rate))))
        self.prototype_threshold.clamp_(.2, 1.5)


class Encoder(nn.Module):
    def __init__(self, width=1.0, hidden=2048, bottleneck=256, prototypes=1024):
        super().__init__()
        self.backbone = backbone(width)
        self.head = DinoHead(int(512 * width), hidden, bottleneck, prototypes)

    def encode(self, x):
        return self.backbone(x)

    def forward(self, x):
        h = self.encode(x)
        return h, self.head(h)


class StrictSpikeEncoder(nn.Module):
    def __init__(self, width=1.0, hidden=2048, prototypes=1024,
                 input_coding='occupancy', population=1):
        super().__init__()
        if input_coding not in ('occupancy', 'bernoulli'):
            raise ValueError(f'Unknown input spike coding: {input_coding}')
        self.input_coding = input_coding
        self.backbone = StrictSpikeBackbone(width)
        self.head = StrictSpikeDinoHead(int(512 * width), hidden, prototypes, population)
        self.last_rates = {}

    def encode(self, x):
        # Both choices preserve polarity and time bins. Bernoulli coding
        # preserves intensity in expectation and avoids a dense occupancy map.
        input_spikes = ((x > 0).to(dtype=x.dtype) if self.input_coding == 'occupancy'
                        else torch.bernoulli(x.clamp(0, 1)))
        h = self.backbone(input_spikes)
        self.last_rates = {'input_spike_rate': input_spikes.detach().float().mean().item(),
                           'backbone_output_rate': h.detach().float().mean().item()}
        return h

    def forward(self, x):
        h = self.encode(x)
        spikes = self.head(h)
        self.last_rates.update(self.head.last_rates)
        return h, spikes


def independent_forward(model, x):
    # Reset on entry as well as exit, so callers cannot leak a previous view.
    functional.reset_net(model)
    try:
        return model(x)
    finally:
        functional.reset_net(model)


class DinoLoss(nn.Module):
    def __init__(self, prototypes, student_temp=.1, teacher_temp=.04,
                 center_momentum=.9, population=1):
        super().__init__()
        if min(student_temp, teacher_temp) <= 0 or not 0 <= center_momentum < 1:
            raise ValueError('Invalid temperature or center momentum')
        self.register_buffer('center', torch.zeros(1, prototypes))
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.population = population

    def rate_logits(self, logits):
        values = logits.float()
        if self.population > 1:
            if values.shape[-1] != self.center.shape[-1] * self.population:
                raise ValueError('Prototype population shape does not match the DINO center')
            values = values.reshape(*values.shape[:-1], self.center.shape[-1], self.population).mean(-1)
        return values.mean(0)

    def teacher_distribution(self, logits):
        return F.softmax((self.rate_logits(logits.detach()) - self.center) /
                         self.teacher_temp, dim=-1)

    def forward(self, student, teacher):
        if len(teacher) != 2 or len(student) < 2:
            raise ValueError('First two views must be matching global views')
        # Aggregate continuous logits before softmax, not probabilities.
        s = [F.log_softmax(self.rate_logits(z) / self.student_temp, -1) for z in student]
        t = [self.teacher_distribution(z) for z in teacher]
        terms = [-(p * logq).sum(-1).mean()
                 for i, p in enumerate(t) for j, logq in enumerate(s) if i != j]
        return torch.stack(terms).mean()

    @torch.no_grad()
    def update_center(self, teacher):
        value = torch.cat([self.rate_logits(z.detach()) for z in teacher]).mean(0, keepdim=True)
        self.center.lerp_(value, 1 - self.center_momentum)


class Experiment(nn.Module):
    def __init__(self, method='dino', width=1., hidden=2048, bottleneck=256,
                 prototypes=1024, projection_dim=128, teacher_momentum=.996,
                 output_target_rate=None, input_coding='occupancy', population=1,
                 teacher_input_coding=None):
        super().__init__()
        if method not in ('dino', 'spike_dino', 'spikeclr', 'temporal_ntxent'):
            raise ValueError(method)
        if not 0 <= teacher_momentum <= 1:
            raise ValueError('Invalid teacher momentum')
        if output_target_rate is not None and (method != 'spike_dino' or not 0 < output_target_rate < 1):
            raise ValueError('Output rate control is only for strict Spike-DINO and needs a target in (0,1)')
        if input_coding != 'occupancy' and method != 'spike_dino':
            raise ValueError('Input spike coding is only configurable for strict Spike-DINO')
        if population < 1 or (population != 1 and method != 'spike_dino'):
            raise ValueError('Population coding is only configurable for strict Spike-DINO')
        if teacher_input_coding is not None and (method != 'spike_dino' or
                                                 teacher_input_coding not in ('occupancy','bernoulli')):
            raise ValueError('Teacher coding is only configurable for strict Spike-DINO')
        self.method = method
        self.teacher_momentum = teacher_momentum
        self.output_target_rate = output_target_rate
        if method in ('dino', 'spike_dino'):
            self.student = (StrictSpikeEncoder(width, hidden, prototypes, input_coding, population)
                            if method == 'spike_dino'
                            else Encoder(width, hidden, bottleneck, prototypes))
            self.teacher = deepcopy(self.student).requires_grad_(False)
            if teacher_input_coding is not None:
                self.teacher.input_coding = teacher_input_coding
            self.criterion = DinoLoss(prototypes, population=population)
        else:
            self.student = EventCLR(backbone(width), int(512 * width), projection_dim,
                                    tau=2., v_reset=0., detach_reset=True,
                                    surrogate_function=surrogate.ATan(), backend='torch')
            adapt_projection_bn(self.student)
            self.teacher = None
            self.criterion = TemporalNTXentLoss(
                temperature=.2, strategy='naive' if method == 'spikeclr' else 'temporal')

    def forward(self, views):
        if len(views) < 2 or (self.teacher is None and len(views) != 2):
            raise ValueError('Contrastive baselines need exactly two views; DINO needs at least two')
        if any(v.ndim != 5 or v.shape[2] != 2 for v in views):
            raise ValueError('Expected [T,B,2,H,W] event views')
        teacher_logits = []
        if self.teacher is not None:
            # Train mode: teacher uses per-view batch statistics, without gradients.
            # Eval mode: uses buffers copied from the student after each update.
            with torch.no_grad():
                teacher_logits = [independent_forward(self.teacher, v)[1] for v in views[:2]]
        pairs = [independent_forward(self.student, v) for v in views]
        logits = [p[1] for p in pairs]
        loss = (self.criterion(logits, teacher_logits) if self.teacher is not None
                else self.criterion(*logits))
        return loss, pairs[0][0].detach().mean(0), teacher_logits

    @torch.no_grad()
    def after_optimizer_step(self, teacher_logits):
        if self.teacher is None:
            return
        if self.output_target_rate is not None:
            self.student.head.adjust_output_threshold(self.output_target_rate)
        m = self.teacher_momentum
        for target, source in zip(self.teacher.parameters(), self.student.parameters()):
            target.lerp_(source, 1 - m)
        # Explicit BN policy: copy running statistics and integer counters.
        for target, source in zip(self.teacher.buffers(), self.student.buffers()):
            target.copy_(source)
        self.criterion.update_center(teacher_logits)


@torch.no_grad()
def diagnostics(features, teacher_logits):
    x = features.float()
    result = {'feature_std': x.std(0, unbiased=False).mean().item()}
    singular = torch.linalg.svdvals(x - x.mean(0))
    probabilities = singular / singular.sum().clamp_min(1e-12)
    result['effective_rank'] = (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp().item() if singular.sum() > 0 else 0.
    if teacher_logits:
        # Uncentered logits summary; objective-specific entropy logged in runner.
        result['teacher_logit_std'] = torch.cat(teacher_logits, dim=1).float().std(unbiased=False).item()
    return result
