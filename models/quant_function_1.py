import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Function
import torch
import numpy as np
import matplotlib.pyplot as plt
class ReScaWConv(nn.Module):
    def __init__(self, in_chn, out_chn, num_bits, kernel_size=3, stride=1, padding=1):
        super(ReScaWConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.num_bits = num_bits
        self.kernel_size = (kernel_size,kernel_size)
        self.out_channels = out_chn
        self.bias = None
        self.clip_val = nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        self.zero = nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = nn.Parameter((torch.rand(self.shape)-0.5) * 0.001, requires_grad=True)

    def forward(self, x):

        real_weights = self.weight
        gamma = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,
                       keepdim=True)
        gamma = gamma.detach()
        scaled_weights = real_weights/gamma
        cliped_weights = torch.clamp(scaled_weights,-1,1)
        n = float(2 ** self.num_bits - 1) / self.clip_val
        quan_weights_no_grad = gamma * (torch.round((cliped_weights + self.clip_val/2) * n ) / n - self.clip_val/2)
        quan_weights = quan_weights_no_grad.detach() - scaled_weights.detach() + scaled_weights
        y = F.conv2d(x, quan_weights, stride=self.stride, padding=self.padding)

        return y


# 8-bit quantization for the first and the last layer
class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                         bias)

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - self.weight).detach() + self.weight
        return F.conv2d(x, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - self.weight).detach() + self.weight
        return F.linear(x, weight_q, self.bias)