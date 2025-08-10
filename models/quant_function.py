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
        # 量化参数
        self.clip_val = nn.Parameter(torch.Tensor([2.0]), requires_grad=False)# self.clip_val = 2.0
        self.zero = nn.Parameter(torch.Tensor([0]), requires_grad=False)# self.zero = 0
       # 虽然在当前代码中zero没有被显式使用
        # 但它通常用于：
        # 1. 非对称量化的零点偏移
        # 2. 量化参数的完整性
        # 3. 未来扩展的预留参数

        # 权重初始化（32位全精度）
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = nn.Parameter((torch.rand(self.shape)-0.5) * 0.001, requires_grad=True)
        self.weight_q = nn.Parameter(torch.rand(self.shape), requires_grad=True)

    def forward(self, x):
        # 重缩放前：权重可能分布在[-0.2, 0.2] 
        real_weights = self.weight
        #论文中提到的三种选择之一：1-norm mean value
        gamma = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,
                       keepdim=True)
        # gamma.detach()：阻止梯度传播到gamma
        gamma = gamma.detach()
        # 重缩放后：权重分布扩展到[-1, 1]，充分利用量化范围
        scaled_weights = real_weights/gamma
        # clamp操作确保权重在[-1,1]范围内
        # 防止异常值影响量化效果
        cliped_weights = torch.clamp(scaled_weights,-1,1)
        # 量化过程：
        # 对于4位量化：
# n = (2^4 - 1) / 2.0 = 15 / 2 = 7.5

# 量化步骤：
# 1. 平移：cliped_weights + 1.0  ([-1,1] → [0,2])
# 2. 缩放：* 7.5               ([0,2] → [0,15]) 
# 3. 量化：round()              (连续值 → 整数 {0,1,2,...,15})
# 4. 反缩放：/ 7.5              ([0,15] → [0,2])
# 5. 反平移：- 1.0              ([0,2] → [-1,1])
# 6. 恢复缩放：* gamma          (恢复原始幅度)
        n = float(2 ** self.num_bits - 1) / self.clip_val
        quan_weights_no_grad = gamma * (torch.round((cliped_weights + self.clip_val/2) * n ) / n - self.clip_val/2)
        # 这是Straight-Through Estimator的精妙实现
# A = quan_weights_no_grad.detach()  # 量化权重，不参与梯度
# B = scaled_weights.detach()        # 原始权重，不参与梯度  
# C = scaled_weights                 # 原始权重，参与梯度计算

# 前向传播：quan_weights = A - B + C = A (使用量化权重)
# 反向传播：∂loss/∂quan_weights = ∂loss/∂C (梯度传给原始权重)
# # 根据链式法则：
# ∂loss/∂quan_weights = ∂loss/∂(A-B+C)
#                      = ∂loss/∂A - ∂loss/∂B + ∂loss/∂C
#                      = 0 - 0 + ∂loss/∂C        # A和B项的梯度被detach()截断
#                      = ∂loss/∂scaled_weights   # 梯度完全传给原始权重！                                
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