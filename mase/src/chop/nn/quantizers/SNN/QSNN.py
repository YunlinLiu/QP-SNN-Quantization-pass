import torch
import torch.nn as nn


class QSNN(nn.Module):
    def __init__(self):
        super(QSNN, self).__init__()

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        if weights.dim() == 4:
            alpha = weights.abs().mean(dim=(1, 2, 3), keepdim=True).detach()
        elif weights.dim() == 3:
            alpha = weights.abs().mean(dim=(1, 2), keepdim=True).detach()
        else:
            alpha = weights.abs().mean(dim=1, keepdim=True).detach()

        alpha = torch.clamp(alpha, min=1e-8)
        w_bin = alpha * torch.sign(weights)
        w_ste = w_bin.detach() - weights.detach() + weights
        return w_ste


