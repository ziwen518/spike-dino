"""Numeric precision controls for unusually large SNN surrogate gradients."""

import torch


class FP64SecondMomentAdamW(torch.optim.AdamW):
    """AdamW that retains fp64 exp_avg_sq across state_dict reloads."""

    def load_state_dict(self, state_dict):
        # PyTorch's base loader unconditionally casts floating state to the
        # parameter dtype, which would overflow fp64 moments during resume.
        saved_ids = [identifier for group in state_dict['param_groups']
                     for identifier in group['params']]
        parameters = [parameter for group in self.param_groups
                      for parameter in group['params']]
        moments = {
            identifier: state['exp_avg_sq']
            for identifier, state in state_dict['state'].items()
            if 'exp_avg_sq' in state
        }
        sanitized = {
            'state': {
                identifier: {
                    **state,
                    'exp_avg_sq': torch.zeros_like(state['exp_avg_sq'],
                                                   dtype=torch.float32),
                } if 'exp_avg_sq' in state else state
                for identifier, state in state_dict['state'].items()
            },
            'param_groups': state_dict['param_groups'],
        }
        result = super().load_state_dict(sanitized)
        for identifier, parameter in zip(saved_ids, parameters, strict=True):
            if identifier in moments:
                self.state[parameter]['exp_avg_sq'] = moments[identifier].to(
                    dtype=torch.float64, device=parameter.device)
        return result


def use_fp64_adam_second_moment(optimizer):
    """Keep AdamW's squared-gradient accumulator in fp64.

    This preserves the optimizer equation when finite fp32 gradients exceed
    sqrt(fp32 max). It does not reduce a competing loss gradient or alter
    the forward SNN. Call after construction and again after loading a state.
    The ordinary AdamW foreach path cannot mix fp32/fp64 state tensors.
    """
    if not isinstance(optimizer, FP64SecondMomentAdamW):
        raise TypeError('Expected FP64SecondMomentAdamW')
    for group in optimizer.param_groups:
        if group.get('foreach') is not False:
            raise ValueError('fp64 AdamW moments require foreach=False')
        for parameter in group['params']:
            state = optimizer.state[parameter]
            if not state:
                state['step'] = torch.zeros((), dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(parameter)
                state['exp_avg_sq'] = torch.zeros_like(parameter,
                                                       dtype=torch.float64)
            else:
                if not torch.isfinite(state['exp_avg_sq']).all():
                    raise ValueError('Cannot recover non-finite AdamW second moment')
                state['exp_avg_sq'] = state['exp_avg_sq'].to(torch.float64)
