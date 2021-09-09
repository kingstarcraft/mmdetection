from ..builder import BACKBONES
from torch import nn


@BACKBONES.register_module()
class NLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3, kernel_size=3, padding=1):
        super(NLayer, self).__init__()
        net = [nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=1, padding=padding
        ), nn.LeakyReLU(0.2, True)]
        scale = 1
        for n in range(1, num_layers):
            net += [
                nn.Conv2d(int(in_channels * scale), int(in_channels * scale / 2),
                          kernel_size=kernel_size, stride=2, padding=padding, bias=False),
                nn.BatchNorm2d(int(in_channels * scale / 2)),
                nn.LeakyReLU(0.2, True)
            ]
            scale = scale / 2
        net += [
            nn.Conv2d(
                int(in_channels * scale), out_channels,
                kernel_size=kernel_size, stride=1, padding=padding)
        ]
        self._net = nn.Sequential(*net)

    def forward(self, x):
        return self._net(x)
