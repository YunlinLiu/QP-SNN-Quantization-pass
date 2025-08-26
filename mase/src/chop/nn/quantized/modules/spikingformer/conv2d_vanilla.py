import torch
from torch import nn

from chop.nn.quantizers.SNN.Vanilla import Vanilla


class Conv2dVanilla(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', config=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                        dilation, groups, bias, padding_mode)
        self.weight_quan = Vanilla(num_bits=config["num_bits"])

    def forward(self, x):
        quantized_weight = self.weight_quan(self.weight)
        return self._conv_forward(x, quantized_weight, self.bias)


