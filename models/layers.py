import torch
import torch.nn as nn


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor): # TBCHW
        y_shape = [x_seq.shape[0], x_seq.shape[1]]  #T*B,C,H,W
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class ClassifyLinear(nn.Module):

    def __init__(self, linear, ):
        super(ClassifyLinear, self).__init__()
        self.ops = linear

    def forward(self, x):
        step = x.size(1)
        out = []
        for i in range(step):
            out += [self.ops(x[:,i,:])]
        out = torch.stack(out,dim=1)
        return out

class Layer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply    #ZIF = Zero If Function，这是作者自己实现的脉冲激活函数！
        # ZIF.apply是PyTorch调用自定义Function的标准方式
        self.thresh = thresh
        self.tau = tau
        self.gama = gama

    def forward(self, x):
        mem = 0#每个batch的mem都初始化为0
        spike_pot = []#每个batch的spike_pot都初始化为空列表
        T = x.shape[1]#T是时间步数，T=4
        # 🔥 关键循环：4次膜电位更新！
        for t in range(T):
            if len(x.shape)==3:
                inp = x[:,t,:]
            else:
                # 第1步：取第t个时间步的输入
                inp = x[:,t,:,:,:] # (256, 64, 32, 32)
            # 第2步：更新膜电位
            mem = mem * self.tau + inp     
            # 第3步：脉冲发放
            spike = self.act(mem - self.thresh, self.gama)
            # 第4步：膜电位重置（硬重置）
            mem = (1 - spike) * mem
            # 第5步：保存这个时间步的脉冲
            spike_pot.append(spike)# spike形状: (256, 64, 32, 32)
             # 🎯 合并所有时间步的脉冲
        return torch.stack(spike_pot, dim=1)# (256, 4, 64, 32, 32)


def add_dimention(x, T):
    x.unsqueeze_(1) # 在第1维插入新维度
    x = x.repeat(1, T, 1, 1, 1)# 在时间维度上重复T次
    return x
# 输入：标准的4维图像数据
#nput_shape = (256, 3, 32, 32)  # (Batch, Channel, Height, Width)

# 步骤1：unsqueeze_(1) 在第1维插入时间维度
#after_unsqueeze = (256, 1, 3, 32, 32)  # (B, T=1, C, H, W)

# 步骤2：repeat(1, T, 1, 1, 1) 在时间维度重复
#final_shape = (256, 4, 3, 32, 32)      # (B, T=4, C, H, W)

# ----- For ResNet19 code -----


class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        x_ = self.layer(x)
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_

class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)# 标准批归一化
        self.seqbn = SeqToANNContainer(self.bn)# 2. 包装成支持时间序列的版本

    def forward(self, x):
        y = self.seqbn(x)
        return y

# # cla params
# class tdBatchNorm(nn.Module):
#     def __init__(self, out_panel):
#         super(tdBatchNorm, self).__init__()
#         self.seqbn = SeqToANNContainer(nn.BatchNorm2d(out_panel))
#
#     def forward(self, x):
#         y = self.seqbn(x)
#         return y

"""class myBatchNorm3d(nn.Module):
    def __init__(self, inplanes, step):
        super().__init__()
        self.bn = nn.BatchNorm3d(inplanes)
        self.step = step
    def forward(self, x):
        out = x.permute(1, 2, 0, 3, 4)
        out = self.bn(out)
        out = out.permute(2, 0, 1, 3, 4).contiguous()
        return out"""