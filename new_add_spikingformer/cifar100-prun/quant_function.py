import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Function
import torch
import numpy as np
import matplotlib.pyplot as plt
class ReScaWConv2d(nn.Module):
    def __init__(self, in_chn, out_chn, num_bits=4, kernel_size=3, stride=1, padding=0):
        super(ReScaWConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.num_bits = num_bits
        self.kernel_size = (kernel_size,kernel_size)
        self.out_channels = out_chn
        self.bias = None
        init_act_clip_val = 2.0
        # self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]), requires_grad=True)
        # self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]), requires_grad=False)
        self.clip_val = torch.Tensor([init_act_clip_val]).cuda()
        # self.zero = nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.zero = torch.Tensor([0]).cuda()
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = nn.Parameter((torch.rand(self.shape)-0.5) * 0.001, requires_grad=True)

    def forward(self, x):

        real_weights = self.weight
        gamma = (2**self.num_bits - 1)/(2**(self.num_bits - 1)) #
        scaling_factor = gamma * torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        # scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        scaled_weights = real_weights/scaling_factor
        cliped_weights = torch.clamp(scaled_weights,-1,1)
        n = float(2 ** self.num_bits - 1) / self.clip_val
        quan_weights_no_grad = scaling_factor * (torch.round((cliped_weights + self.clip_val/2) * n ) / n - self.clip_val/2)
        # quan_weights_no_grad = torch.round((cliped_weights + self.clip_val / 2) * n) / n - self.clip_val / 2
        quan_weights = quan_weights_no_grad.detach() - scaled_weights.detach() + scaled_weights
        y = F.conv2d(x, quan_weights, stride=self.stride, padding=self.padding)

        return y

class ReScaWConv1d(nn.Module):
    def __init__(self, in_chn, out_chn, num_bits=4, kernel_size=3, stride=1, padding=0):
        super(ReScaWConv1d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.num_bits = num_bits
        self.kernel_size = kernel_size
        self.out_channels = out_chn
        self.bias = None
        init_act_clip_val = 2.0
        # self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]), requires_grad=True)
        # self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]), requires_grad=False)
        self.clip_val = torch.Tensor([init_act_clip_val]).cuda()
        # self.zero = nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.zero = torch.Tensor([0]).cuda()
        self.shape = (out_chn, in_chn, kernel_size)
        self.weight = nn.Parameter((torch.rand(self.shape)-0.5) * 0.001, requires_grad=True)

    def forward(self, x):

        real_weights = self.weight
        gamma = (2**self.num_bits - 1)/(2**(self.num_bits - 1)) #
        scaling_factor = gamma * torch.mean(torch.mean(abs(real_weights),dim=2,keepdim=True),dim=1,keepdim=True)
        # scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        scaled_weights = real_weights/scaling_factor
        cliped_weights = torch.clamp(scaled_weights,-1,1)
        n = float(2 ** self.num_bits - 1) / self.clip_val
        quan_weights_no_grad = scaling_factor * (torch.round((cliped_weights + self.clip_val/2) * n ) / n - self.clip_val/2)
        # quan_weights_no_grad = torch.round((cliped_weights + self.clip_val / 2) * n) / n - self.clip_val / 2
        quan_weights = quan_weights_no_grad.detach() - scaled_weights.detach() + scaled_weights
        y = F.conv1d(x, quan_weights, stride=self.stride, padding=self.padding)

        return y


