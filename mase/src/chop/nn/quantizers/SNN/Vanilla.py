import torch
import torch.nn as nn


class Vanilla(nn.Module):
    """Vanilla uniform quantizer for weight quantization (no rescaling)."""

    def __init__(self, num_bits, clip_val=2.0):
        super(Vanilla, self).__init__()
        self.num_bits = num_bits
        self.clip_val = nn.Parameter(torch.Tensor([clip_val]), requires_grad=False)

    def forward(self, weights):
        """Forward pass for vanilla uniform weight quantization."""
        if self.clip_val.device != weights.device:
            self.clip_val = self.clip_val.to(weights.device)

        real_weights = weights

        # Clip weights to [-1, 1]
        clipped_weights = torch.clamp(real_weights, -1, 1)

        # Quantization step: n = (2^b - 1) / clip_val
        n = float(2 ** self.num_bits - 1) / self.clip_val

        # Uniform quantization in [-1, 1]
        quan_weights_no_grad = (
            torch.round((clipped_weights + self.clip_val / 2) * n) / n - self.clip_val / 2
        )

        # Straight-through estimator
        quan_weights = quan_weights_no_grad.detach() - real_weights.detach() + real_weights

        return quan_weights


