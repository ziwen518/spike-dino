import torch
from torch.nn import functional as F
from spikingjelly.activation_based import functional

from spike_dino.dinov3 import (SpikingProjectionHead, masked_patch_cross_entropy,
                               sinkhorn_knopp)
from spike_dino.pretrain_v3 import fixed_count_masks


torch.set_num_threads(2)


def test_ibot_reduction_matches_per_sample_reference_and_gradients():
    torch.manual_seed(24)
    mask = torch.tensor([[True, False, False], [True, True, True],
                         [False, False, False], [False, True, False]])
    logits = torch.randn(4, 3, 5, requires_grad=True)
    targets = torch.randn(4, 3, 5).softmax(-1)
    token_ce = -(targets * F.log_softmax(logits / .1, -1)).sum(-1)
    reference = ((token_ce * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
    actual = masked_patch_cross_entropy(logits[mask], targets[mask],
                                        [mask[:2], mask[2:]])
    torch.testing.assert_close(actual, reference)
    a, = torch.autograd.grad(actual, logits, retain_graph=True)
    b, = torch.autograd.grad(reference, logits)
    torch.testing.assert_close(a, b)
    assert not a[~mask].any()


def test_linear_readout_preserves_learnable_weight_scale():
    torch.manual_seed(25)
    head = SpikingProjectionHead(16, 32, 8, 12, prototype_readout='linear')
    x = (torch.rand(4, 6, 16) > .5).float()
    original = head(x).detach()
    assert original.abs().max() > 0
    functional.reset_net(head)
    with torch.no_grad():
        head.prototypes.mul_(2)
    torch.testing.assert_close(head(x), original * 2)
    head.prototype_readout = 'cosine'
    functional.reset_net(head)
    original = head(x).detach()
    functional.reset_net(head)
    with torch.no_grad():
        head.prototypes.mul_(2)
    torch.testing.assert_close(head(x), original)


def test_sinkhorn_matches_probability_domain_at_normal_scale_and_stays_balanced():
    torch.manual_seed(26)
    logits = torch.randn(8, 128) * .06
    q = (logits / .04).exp().T
    q /= q.sum()
    for _ in range(3):
        q /= q.sum(1, keepdim=True)
        q /= q.shape[0]
        q /= q.sum(0, keepdim=True)
        q /= q.shape[1]
    reference = (q * q.shape[1]).T
    torch.testing.assert_close(sinkhorn_knopp(logits), reference, atol=1e-6, rtol=1e-5)
    large = sinkhorn_knopp(torch.randn(8, 65536) * 1.0)
    assert torch.isfinite(large).all()
    assert (large.mean(0) > 0).all()
    torch.testing.assert_close(large.sum(-1), torch.ones(8), atol=2e-5, rtol=2e-5)


def test_fixed_count_masks_stratify_global_samples_and_reproduce():
    torch.manual_seed(27)
    first = fixed_count_masks(4, 256, .1, .5, 'cpu', .5)
    torch.manual_seed(27)
    second = fixed_count_masks(4, 256, .1, .5, 'cpu', .5)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)
    counts = torch.cat(first).sum(-1)
    assert int((counts > 0).sum()) == 4
    assert sorted(counts[counts > 0].tolist()) == [51, 76, 102, 128]
