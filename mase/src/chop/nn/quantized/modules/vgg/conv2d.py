import torch
from torch import nn

from chop.nn.quantizers.SNN.ReScaW import ReScaW


class Conv2dReScaW(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', config=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                        dilation, groups, bias, padding_mode)
        # NOTE: The only change from the original Conv2d is the quantization of weights
        # Preserving the original layer architecture for state_dict compatibility
        # clip_val is fixed to 2.0 in ReScaW algorithm, so we don't need to pass it
        self.weight_quan = ReScaW(num_bits=config["num_bits"])

    def forward(self, x):
        quantized_weight = self.weight_quan(self.weight)
        return self._conv_forward(x, quantized_weight, self.bias)
