import torch
import torch.nn as nn


class ReScaW(nn.Module):
    """ReScaW quantizer for weight quantization."""
    
    def __init__(self, num_bits, clip_val=2.0):
        super(ReScaW, self).__init__()
        self.num_bits = num_bits
        self.clip_val = nn.Parameter(torch.Tensor([clip_val]), requires_grad=False)
    
    def forward(self, weights):
        """Forward pass for weight quantization."""
        if self.clip_val.device != weights.device:
            self.clip_val = self.clip_val.to(weights.device)
            
        real_weights = weights
        
        # Calculate channel-wise scaling factor (gamma)
        gamma = torch.mean(
            torch.mean(
                torch.mean(abs(real_weights), dim=3, keepdim=True), 
                dim=2, keepdim=True
            ), 
            dim=1, keepdim=True
        )
        gamma = gamma.detach()
        
        # Scale weights by gamma
        scaled_weights = real_weights / gamma
        
        # Clip scaled weights to [-1, 1]
        clipped_weights = torch.clamp(scaled_weights, -1, 1)
        
        # Quantization step
        n = float(2 ** self.num_bits - 1) / self.clip_val
        
        # Uniform quantization
        quan_weights_no_grad = gamma * (
            torch.round((clipped_weights + self.clip_val/2) * n) / n - self.clip_val/2
        )
        
        # Straight-through estimator
        quan_weights = quan_weights_no_grad.detach() - scaled_weights.detach() + scaled_weights
        
        return quan_weights