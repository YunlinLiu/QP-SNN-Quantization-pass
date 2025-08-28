import torch
import torch.nn as nn


class APoT(nn.Module):
    def __init__(self, num_bits: int, base_k: int = 2, eps: float = 1e-8):
        super(APoT, self).__init__()
        self.num_bits = num_bits
        self.base_k = base_k
        self.eps = eps

    def _apot_project_abs(self, x_abs: torch.Tensor) -> torch.Tensor:
        b = self.num_bits
        k = self.base_k
        assert b % k == 0 and k >= 1
        n = b // k

        residual = x_abs
        q_sum = torch.zeros_like(x_abs)

        max_m = 2 * k - 2
        for i in range(n - 1, -1, -1):
            options = [torch.zeros_like(x_abs)]
            for m in range(0, max_m + 1):
                e = i + m * n
                options.append(torch.tensor(2.0 ** (-e), device=x_abs.device, dtype=x_abs.dtype))

            diffs = None
            for idx, s in enumerate(options):
                d = torch.abs(residual - s)
                diffs = d if diffs is None else torch.stack([diffs, d], dim=0) if idx == 1 else torch.cat([diffs, d.unsqueeze(0)], dim=0)

            sel = torch.argmin(diffs, dim=0)
            chosen = options[0]
            for idx in range(1, len(options)):
                chosen = torch.where(sel == idx, options[idx], chosen)

            q_sum = q_sum + chosen
            residual = residual - chosen

        return q_sum

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        alpha = weights.detach().abs().amax()
        alpha = torch.clamp(alpha, min=self.eps)

        w = weights / alpha
        w = torch.clamp(w, -1.0, 1.0)

        sign = torch.sign(w)
        x_abs = torch.abs(w)
        q_abs = self._apot_project_abs(x_abs)
        q = sign * q_abs
        q = torch.clamp(q, -1.0, 1.0)

        q = alpha * q
        q_ste = q.detach() - weights.detach() + weights
        return q_ste


