"""Task-balanced gradients for the shared spiking backbone.

This is an experimental optimizer route, not part of the published DINOv3
objective. It changes only backward aggregation; forward spikes and losses
remain untouched.
"""

import torch


def _backbone_group(name):
    parts = name.split('.')
    if len(parts) > 2 and parts[0] == 'backbone' and parts[1] == 'blocks':
        return '.'.join(parts[:3])
    return '.'.join(parts[:2])


def balanced_backbone_gradients(student, global_loss, local_loss,
                                protect_stem=False):
    """Equalize global/local L2 contributions within each backbone block.

    Head parameters retain their original loss gradients. For a shared group,
    each nonzero task gradient is divided by its own group L2 norm before
    summation. This avoids a local objective with a larger surrogate Jacobian
    silently replacing the CLS objective, without a manually chosen weight.
    """
    named = list(student.named_parameters())
    parameters = [parameter for _, parameter in named]
    global_grads = torch.autograd.grad(
        global_loss, parameters, retain_graph=True, allow_unused=True)
    local_grads = torch.autograd.grad(
        local_loss, parameters, retain_graph=False, allow_unused=True)
    groups = {}
    protected = {'backbone.patch_embed', 'backbone.patch_norm',
                 'backbone.position_current', 'backbone.cls_current',
                 'backbone.cls_pool_norm'}
    for index, ((name, _), global_grad, local_grad) in enumerate(
            zip(named, global_grads, local_grads)):
        if name.startswith('backbone.'):
            groups.setdefault(_backbone_group(name), []).append(index)
        else:
            grad = (global_grad if local_grad is None else
                    local_grad if global_grad is None else
                    global_grad + local_grad)
            named[index][1].grad = grad
    diagnostics = {}
    for group, indices in groups.items():
        def norm(grads):
            squares = [grads[index].detach().double().square().sum()
                       for index in indices if grads[index] is not None]
            return torch.stack(squares).sum().sqrt() if squares else None

        global_norm = norm(global_grads)
        local_norm = norm(local_grads)
        global_only = protect_stem and group in protected
        for index in indices:
            global_grad = global_grads[index]
            local_grad = local_grads[index]
            pieces = []
            if global_grad is not None and bool(global_norm > 0):
                pieces.append(global_grad / global_norm.to(global_grad.dtype))
            if (not global_only and local_grad is not None and
                    bool(local_norm > 0)):
                pieces.append(local_grad / local_norm.to(local_grad.dtype))
            named[index][1].grad = sum(pieces) if pieces else None
        diagnostics[group] = {
            'global_norm': float(global_norm) if global_norm is not None else 0.,
            'local_norm': float(local_norm) if local_norm is not None else 0.,
        }
    return diagnostics


def global_preserving_backbone_gradients(student, global_loss, local_loss):
    """Keep the global gradient and cap orthogonalized local block gradients.

    For each backbone group, remove only the component of the local gradient
    that opposes the global one. The remaining local contribution is capped
    at the original global group norm. No relative loss weight is introduced.
    Heads still receive their original task gradients. This is an experimental
    SNN multi-objective backward rule, not part of DINOv3.
    """
    named = list(student.named_parameters())
    parameters = [parameter for _, parameter in named]
    global_grads = torch.autograd.grad(
        global_loss, parameters, retain_graph=True, allow_unused=True)
    local_grads = torch.autograd.grad(
        local_loss, parameters, retain_graph=False, allow_unused=True)
    groups = {}
    for index, ((name, parameter), global_grad, local_grad) in enumerate(
            zip(named, global_grads, local_grads)):
        if name.startswith('backbone.'):
            groups.setdefault(_backbone_group(name), []).append(index)
        else:
            parameter.grad = (global_grad if local_grad is None else
                              local_grad if global_grad is None else
                              global_grad + local_grad)
    diagnostics = {}
    for group, indices in groups.items():
        global_sq = sum((global_grads[index].detach().double().square().sum()
                         for index in indices if global_grads[index] is not None),
                        torch.tensor(0., dtype=torch.float64,
                                     device=parameters[indices[0]].device))
        local_sq = sum((local_grads[index].detach().double().square().sum()
                        for index in indices if local_grads[index] is not None),
                       torch.tensor(0., dtype=torch.float64,
                                    device=parameters[indices[0]].device))
        dot = sum((
            (global_grads[index].detach().double() *
             local_grads[index].detach().double()).sum()
            for index in indices if global_grads[index] is not None and
            local_grads[index] is not None),
            torch.tensor(0., dtype=torch.float64,
                         device=parameters[indices[0]].device))
        # A negative coefficient is the opposing component to remove.
        coefficient = min(float(dot / global_sq), 0.) if global_sq > 0 else 0.
        projected = {}
        for index in indices:
            local = local_grads[index]
            global_grad = global_grads[index]
            if local is None:
                projected[index] = None
            elif global_grad is None or coefficient == 0.:
                projected[index] = local
            else:
                projected[index] = local - coefficient * global_grad
        projected_sq = sum((value.detach().double().square().sum()
                            for value in projected.values() if value is not None),
                           torch.tensor(0., dtype=torch.float64,
                                        device=parameters[indices[0]].device))
        scale = (min(1., float((global_sq / projected_sq).sqrt()))
                 if global_sq > 0 and projected_sq > 0 else 1.)
        for index in indices:
            global_grad = global_grads[index]
            local = projected[index]
            parameters[index].grad = (
                global_grad if local is None else
                local * scale if global_grad is None else
                global_grad + local * scale)
        denominator = (global_sq * local_sq).sqrt()
        diagnostics[group] = {
            'global_norm': float(global_sq.sqrt()),
            'local_norm': float(local_sq.sqrt()),
            'task_cosine': float(dot / denominator) if denominator > 0 else 0.,
            'projected_local_norm': float(projected_sq.sqrt()),
            'local_scale': scale,
        }
    return diagnostics
