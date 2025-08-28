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
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            if len(x.shape)==3:
                inp = x[:,t,:]
            else:
                inp = x[:,t,:,:,:]
            mem = mem * self.tau + inp      # BTCHW C L1
            spike = self.act(mem - self.thresh, self.gama)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)


class LIFSpikeQ(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0, num_bits=2, eps=1e-8):
        super(LIFSpikeQ, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.num_bits = num_bits
        self.eps = eps

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        s = float(2 ** (self.num_bits - 1) - 1) if self.num_bits > 1 else 1.0
        for t in range(T):
            if len(x.shape) == 3:
                inp = x[:, t, :]
            else:
                inp = x[:, t, :, :, :]
            mem = mem * self.tau + inp
            alpha = torch.clamp(mem.detach().abs().amax(), min=self.eps)
            mem_n = torch.clamp(mem / alpha, -1.0, 1.0)
            if self.num_bits > 1:
                mem_q = alpha * (torch.round(mem_n * s) / s)
            else:
                mem_q = alpha * torch.sign(mem_n)
            mem_q = mem_q.detach() - mem.detach() + mem
            spike = self.act(mem_q - self.thresh, self.gama)
            mem = (1 - spike) * mem_q
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)


class MultiStepLIFNodeQ(nn.Module):
    def __init__(self, thresh=1.0, tau=2.0, num_bits=8, gama=1.0, eps=1e-8):
        super(MultiStepLIFNodeQ, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.num_bits = num_bits
        self.eps = eps

    def forward(self, x):
        # x: [T, B, ...]
        T = x.shape[0]
        mem = 0
        spikes = []
        # quant grid scale
        s = float(2 ** (self.num_bits - 1) - 1) if self.num_bits > 1 else 1.0
        for t in range(T):
            inp = x[t]
            # simple decay + integrate; tau acts as decay factor when <=1 or as time-const-like when >1
            if isinstance(self.tau, float) and self.tau <= 1.0:
                mem = mem * self.tau + inp
            else:
                # convert time-constant like to decay factor 1-1/tau
                decay = 1.0 - (1.0 / float(self.tau))
                mem = mem * decay + inp

            # layer-wise alpha_u per step
            alpha = torch.clamp(mem.detach().abs().amax(), min=self.eps)
            mem_n = torch.clamp(mem / alpha, -1.0, 1.0)
            if self.num_bits > 1:
                mem_q = alpha * (torch.round(mem_n * s) / s)
            else:
                mem_q = alpha * torch.sign(mem_n)
            mem_q = mem_q.detach() - mem.detach() + mem

            spike = self.act(mem_q - self.thresh, self.gama)
            mem = (1 - spike) * mem_q
            spikes.append(spike)
        return torch.stack(spikes, dim=0)


def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x


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
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

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