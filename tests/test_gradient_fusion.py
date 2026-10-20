import torch
from torch import nn

from spike_dino.gradient_fusion import (balanced_backbone_gradients,
                                        global_preserving_backbone_gradients)


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.patch_embed = nn.Linear(2, 2, bias=False)
        self.backbone.blocks = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.dino_head = nn.Linear(2, 1, bias=False)
        self.ibot_head = nn.Linear(2, 1, bias=False)


def test_balanced_backbone_gradients_preserve_heads_and_equalize_tasks():
    model = TinyStudent()
    weight = model.backbone.blocks[0].weight
    global_loss = (weight * torch.tensor([[3., 0.], [0., 4.]])).sum()
    local_loss = (weight * torch.tensor([[-6., 0.], [0., 8.]])).sum()
    global_loss = global_loss + 2 * model.dino_head.weight.sum()
    local_loss = local_loss + 3 * model.ibot_head.weight.sum()
    diagnostics = balanced_backbone_gradients(
        model, global_loss, local_loss)
    torch.testing.assert_close(weight.grad,
                               torch.tensor([[0., 0.], [0., 1.6]]))
    torch.testing.assert_close(model.dino_head.weight.grad,
                               torch.full_like(model.dino_head.weight, 2.))
    torch.testing.assert_close(model.ibot_head.weight.grad,
                               torch.full_like(model.ibot_head.weight, 3.))
    assert diagnostics['backbone.blocks.0']['global_norm'] == 5.
    assert diagnostics['backbone.blocks.0']['local_norm'] == 10.


def test_balanced_backbone_gradients_handle_absent_task_gradient():
    model = TinyStudent()
    weight = model.backbone.blocks[0].weight
    global_loss = weight.sum()
    local_loss = model.ibot_head.weight.sum()
    balanced_backbone_gradients(model, global_loss, local_loss)
    torch.testing.assert_close(weight.grad, torch.full_like(weight, .5))


def test_stem_protection_keeps_global_direction_while_balancing_blocks():
    model = TinyStudent()
    stem = model.backbone.patch_embed.weight
    block = model.backbone.blocks[0].weight
    global_loss = stem.sum() + block.sum()
    local_loss = -10 * stem.sum() - 10 * block.sum()
    balanced_backbone_gradients(model, global_loss, local_loss,
                                protect_stem=True)
    torch.testing.assert_close(stem.grad, torch.full_like(stem, .5))
    torch.testing.assert_close(block.grad, torch.zeros_like(block))


def test_global_preserving_projection_removes_conflict_and_caps_local_norm():
    model = TinyStudent()
    weight = model.backbone.blocks[0].weight
    global_direction = torch.tensor([[1., 0.], [0., 0.]])
    local_direction = torch.tensor([[-2., 4.], [0., 0.]])
    global_loss = (weight * global_direction).sum() + 2 * model.dino_head.weight.sum()
    local_loss = (weight * local_direction).sum() + 3 * model.ibot_head.weight.sum()
    diagnostics = global_preserving_backbone_gradients(
        model, global_loss, local_loss)
    torch.testing.assert_close(weight.grad, torch.tensor([[1., 1.], [0., 0.]]))
    torch.testing.assert_close(model.dino_head.weight.grad,
                               torch.full_like(model.dino_head.weight, 2.))
    torch.testing.assert_close(model.ibot_head.weight.grad,
                               torch.full_like(model.ibot_head.weight, 3.))
    block = diagnostics['backbone.blocks.0']
    assert block['task_cosine'] < 0
    assert block['local_scale'] == .25
    assert float((weight.grad * global_direction).sum()) >= float(
        global_direction.square().sum())


def test_global_preserving_projection_keeps_aligned_and_absent_gradients():
    model = TinyStudent()
    block = model.backbone.blocks[0].weight
    stem = model.backbone.patch_embed.weight
    global_loss = 2 * block.sum() + stem.sum()
    local_loss = .5 * block.sum() + model.ibot_head.weight.sum()
    diagnostics = global_preserving_backbone_gradients(
        model, global_loss, local_loss)
    torch.testing.assert_close(block.grad, torch.full_like(block, 2.5))
    torch.testing.assert_close(stem.grad, torch.ones_like(stem))
    assert diagnostics['backbone.blocks.0']['local_scale'] == 1.
