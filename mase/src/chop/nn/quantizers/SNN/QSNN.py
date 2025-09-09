import torch
import torch.nn as nn


class QSNN(nn.Module):
    def __init__(self):
        super(QSNN, self).__init__()

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        # WS-DR weight regulation: layer-wise standardization, channel-wise scaling
        eps = 1e-8
        mu = weights.mean()
        sigma = torch.clamp(weights.std(unbiased=False), min=eps)
        w_hat = (weights - mu) / sigma

        if weights.dim() == 4:
            reduce_dims = (1, 2, 3)
        elif weights.dim() == 3:
            reduce_dims = (1, 2)
        else:
            reduce_dims = (1,)

        alpha = w_hat.abs().mean(dim=reduce_dims, keepdim=True).detach()
        alpha = torch.clamp(alpha, min=eps)
        w_bin = alpha * torch.sign(w_hat)
        # STE: forward uses binarized, backward w.r.t original full-precision weights
        w_ste = w_bin.detach() - weights.detach() + weights
        return w_ste


