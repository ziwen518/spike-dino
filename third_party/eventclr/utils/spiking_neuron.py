from spikingjelly.activation_based import neuron
from torch import nn


def get_spiking_neuron(neuron_type: str, **neuron_kwargs):
    if neuron_type.upper() == 'LIF':
        return neuron.LIFNode(**neuron_kwargs)
    elif neuron_type.upper() == 'IF':
        return neuron.IFNode(**neuron_kwargs)
    elif neuron_type.upper() == 'PLIF':
        return neuron.ParametricLIFNode(**neuron_kwargs)
    elif neuron_type.upper() == 'RELU':
        # ANN-equivalent: plain ReLU activation instead of a spiking neuron.
        # SNN-only kwargs (surrogate_function, decay_input, v_threshold, ...) don't apply here.
        return nn.ReLU(inplace=True)
    else:
        raise ValueError(f"Unsupported neuron type: {neuron_type}")