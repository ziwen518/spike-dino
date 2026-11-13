"""SEW-ResNet-18 spiking backbone for Spike-DINO (the SpikeCLR-matched family).

Why this line exists: our all-spiking ViT learns a DINO objective but its frozen
features barely move, while SpikeCLR's SEW-ResNet-18 trains with a plain
supervised/SimCLR objective at CIFAR10-DVS scale. The two differences that
matter mechanistically are (a) an identity (SEW) residual instead of a
non-negative low-rank count attention and (b) convolution+GAP locality instead
of long-range spiking attention. This module swaps only the backbone so that the
DINO objective, head, views and probe protocol stay bit-comparable with the ViT
line; every tensor entering a synaptic layer is the output of a LIF neuron.

Deviations from the ViT line, all recorded in the run config:
* the SEW-ADD shortcut sums the block spikes onto the identity, so the residual
  stream carries small non-negative *integer counts* (the standard SEW
  formulation, not real-valued activations); every LIF output stays binary;
* input keeps the same 2-channel polarity event histogram (the first conv is
  built for 2 channels instead of ImageNet's 3; nothing is pretrained);
* there is no class token, so the spike readout of a token position is replaced
  by GAP over the layer-4 map (this is exactly SpikeCLR's embedding);
* the frozen probe reports layer-4 GAP ('cls'), layer-3 GAP ('gap') and their
  concatenation ('concat'), plus the same pair of pre-final-LIF BN currents.
"""

import torch
from torch import nn
from spikingjelly.activation_based import functional, layer as slayer, neuron, surrogate


def _lif_kwargs(tau, threshold):
    return dict(tau=tau, v_threshold=threshold, v_reset=0., detach_reset=True,
                surrogate_function=surrogate.ATan(), step_mode='s')


def _store_current(module, _inputs, output):
    """Forward hook: keep the BN current entering a block's final LIF."""
    module.sdino_current = output


class SEWResNetBackbone(nn.Module):
    """spikingjelly ``sew_resnet18`` in multi-step mode ``[T, N, C, H, W]``.

    Interface matches ``SpikingVisionTransformer``: ``forward(events)`` returns
    ``(cls [T,N,512], patch [T,N,tokens,512])`` so the shared encoder code works
    unchanged. ``keep_output_current`` additionally records the BN currents
    entering the final LIF of layer 3 and layer 4, and the layer-3 spike map.
    """

    def __init__(self, *, in_channels=2, neuron_tau=2., neuron_threshold=1.,
                 cnf='ADD'):
        super().__init__()
        from spikingjelly.activation_based.model import sew_resnet
        kwargs = _lif_kwargs(neuron_tau, neuron_threshold)
        # cnf='ADD' is the spike-element-wise identity shortcut of Fang et al.
        # (and of SpikeCLR); spikingjelly also offers 'AND'/'IAND'.
        model = sew_resnet.sew_resnet18(pretrained=False, progress=False, cnf=cnf,
                                        spiking_neuron=neuron.LIFNode, **kwargs)
        # 2-channel polarity histogram instead of 3-channel RGB frames; the
        # network is trained from scratch so no pretrained kernel is lost.
        model.conv1 = slayer.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                    padding=3, bias=False, step_mode='m')
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
        functional.set_step_mode(model, 'm')
        self.model = model
        self.dim = 512
        self.mid_dim = 256
        self.keep_output_current = False
        self.output_current = None
        self.aux = None
        # The hook stores on the module it is called with, never on a closure
        # over this backbone: probe_v4 probes a deepcopy of the encoder, and a
        # hook bound to the original instance would fill the original's cache,
        # leaving the copy's reading None (this happened: AttributeError on
        # 'NoneType'.float in the first SEW probe).
        model.layer3[-1].bn2.register_forward_hook(_store_current)
        model.layer4[-1].bn2.register_forward_hook(_store_current)

    def forward(self, events):
        model = self.model
        if events.shape[2] != model.conv1.in_channels:
            raise ValueError('Event channels do not match the first convolution')
        x = model.sn1(model.bn1(model.conv1(events)))
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        mid = model.layer3(x)
        fine = model.layer4(mid)
        bn_mid, bn_fine = model.layer3[-1].bn2, model.layer4[-1].bn2
        current_fine, current_mid = getattr(bn_fine, 'sdino_current', None),             getattr(bn_mid, 'sdino_current', None)
        # Release immediately: a retained reference would keep the autograd
        # graph of the whole step alive.
        bn_fine.sdino_current = bn_mid.sdino_current = None
        self.aux = self.output_current = None
        if self.keep_output_current:
            self.aux = (mid, current_fine, current_mid)
            self.output_current = (current_fine, current_mid)
        cls = fine.float().mean((3, 4))  # [T,N,C,H,W] -> GAP over H,W
        patch = fine.flatten(3).transpose(2, 3).to(fine.dtype)
        return cls, patch


def sew_representation(encoder, events):
    """Frozen downstream features for the SEW backbone (no head, no labels)."""
    backbone = encoder.backbone
    keep = backbone.keep_output_current
    backbone.keep_output_current = True
    try:
        cls, _patch, _pooled = encoder.encode(events)
        mid, current_fine, current_mid = backbone.aux
    finally:
        backbone.aux = None
        backbone.output_current = None
        backbone.keep_output_current = keep
    # Both maps arrive as [T,N,C,H,W]: average over time and space, keep C.
    fine = cls.float().mean(0)
    middle = mid.float().mean((0, 3, 4))
    return {'cls': fine, 'gap': middle, 'concat': torch.cat((fine, middle), -1),
            'concat_current': torch.cat((current_fine.float().mean((0, 3, 4)),
                                         current_mid.float().mean((0, 3, 4))), -1)}


def sew_classifier_features(encoder, events):
    """Supervised-FT input: layer-4 GAP ++ layer-3 GAP spikes (768), grads on."""
    backbone = encoder.backbone
    keep = backbone.keep_output_current
    backbone.keep_output_current = True
    try:
        cls, _patch, _pooled = encoder.encode(events)
        mid, _current_fine, _current_mid = backbone.aux
    finally:
        backbone.aux = None
        backbone.output_current = None
        backbone.keep_output_current = keep
    return torch.cat((cls.float().mean(0), mid.float().mean((0, 3, 4))), -1)
