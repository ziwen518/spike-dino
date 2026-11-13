import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional

from third_party.eventclr.utils.spiking_neuron import get_spiking_neuron

def sew_function(x: torch.Tensor, y: torch.Tensor, cnf: str):
    if cnf == 'ADD':
        return x + y
    elif cnf == 'AND':
        return x * y
    elif cnf == 'IAND':
        return x * (1. - y)
    elif cnf == 'NONE':
        return y
    else:
        raise NotImplementedError

def get_conv_layer(separable: bool):
    if separable:
        return SeparableConv2d
    return layer.Conv2d


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = layer.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = layer.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SEWBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, separable=False, cnf='ADD', neuron_type='LIF', **kwargs):
        super().__init__()
        self.conv1 = get_conv_layer(separable=separable)(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(out_channels)
        self.sn1 = get_spiking_neuron(neuron_type, **kwargs)

        self.conv2 = get_conv_layer(separable=separable)(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(out_channels)
        self.sn2 = get_spiking_neuron(neuron_type, **kwargs)

        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = get_spiking_neuron(neuron_type, **kwargs)

        self.cnf = cnf


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = sew_function(identity, out, self.cnf)

        return out


class SEWResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            channels,
            strides,
            in_channels=2,
            width=1.0,
            pool=False,
            separable=False,
            cnf='ADD',
            neuron_type='LIF',
            **neuron_kwargs
    ):
        """Modified SEW-ResNet backbone inspired by SpikingJelly."""
        super().__init__()

        assert len(channels) == len(layers) == len(strides), "Length of channels, layers, and strides must be equal."

        def c(x):
            return int(x * width)

        self.in_channels = c(64)
        self.cnf = cnf

        self.separable = separable

        self.cnf = cnf
        self.neuron_type = neuron_type
        self.neuron_kwargs = neuron_kwargs

        self.conv1 = get_conv_layer(separable=separable)(in_channels, c(64), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(c(64))
        self.sn1 = get_spiking_neuron(neuron_type, **neuron_kwargs)

        self.pool = None

        if pool:
            self.pool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList([
            self._make_layer(block, c(channels[i]), layers[i], stride=strides[i]) for i in range(len(layers))
        ])

        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))

        functional.set_step_mode(self, step_mode='m')

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, SEWBasicBlock):
                nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                layer.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(out_channels)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample, self.separable, self.cnf, self.neuron_type, **self.neuron_kwargs)]
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, separable=self.separable, cnf=self.cnf, neuron_type=self.neuron_type, **self.neuron_kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)

        if self.pool is not None:
            x = self.pool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)

        return x

class SEWResNet18(SEWResNet):
    def __init__(self, in_channels=2, width=1.0, cnf='ADD', neuron_type='LIF', **neuron_kwargs):
        super().__init__(
            block=SEWBasicBlock,
            layers=[2, 2, 2, 2],
            channels=[64, 128, 256, 512],
            strides=[1, 2, 2, 2],
            in_channels=in_channels,
            width=width,
            cnf=cnf,
            neuron_type=neuron_type,
            pool=True,
            **neuron_kwargs
        )

class SEWResNet18Sep(SEWResNet):
    def __init__(self, in_channels=2, width=1.0, cnf='ADD', neuron_type='LIF', **neuron_kwargs):
        super().__init__(
            block=SEWBasicBlock,
            layers=[2, 2, 2, 2],
            channels=[64, 128, 256, 512],
            strides=[1, 2, 2, 2],
            in_channels=in_channels,
            width=width,
            cnf=cnf,
            neuron_type=neuron_type,
            separable=True,
            pool=True,
            **neuron_kwargs
        )


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def compare_model_parameters():
    """Compare parameter counts across different SEW-ResNet variants."""
    models = {
        'SEWResNet18': SEWResNet18(in_channels=2),
        'SEWResNet18Sep': SEWResNet18Sep(in_channels=2)
    }

    print(f"{'Model':<20} {'Total Params':<15} {'Trainable Params':<15}")
    print("-" * 50)

    for name, model in models.items():
        total, trainable = count_parameters(model)
        print(f"{name:<20} {total:>14,} {trainable:>14,}")

    # Optional: Compare with different widths
    print("\n" + "=" * 50)
    print("Width Comparison for SEWResNet18:")
    print("=" * 50)
    print(f"{'Width':<10} {'Total Params':<15}")
    print("-" * 25)

    for width in [1.0]:
        model = SEWResNet18(in_channels=2, width=width)
        total, _ = count_parameters(model)
        print(f"{width:<10} {total:>14,}")


if __name__ == "__main__":
    compare_model_parameters()
