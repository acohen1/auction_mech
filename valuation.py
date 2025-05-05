from __future__ import annotations
from typing import Dict, Callable
import numpy as np
import torch


class ValuationModel:
    """Base class for bidder-valuation generators."""
    def __init__(self, n_bidders: int):
        self.n = n_bidders

    def sample(self, n_samples: int) -> torch.Tensor:
        raise NotImplementedError


# ------------------ concrete generators ------------------ #

class IIDUniform(ValuationModel):
    def sample(self, n_samples: int) -> torch.Tensor:
        return torch.rand(n_samples, self.n)


class IIDExponential(ValuationModel):
    def __init__(self, n_bidders: int, mean: float = 1.0):
        super().__init__(n_bidders)
        self.scale = mean

    def sample(self, n_samples: int) -> torch.Tensor:
        return torch.from_numpy(
            np.random.exponential(self.scale, size=(n_samples, self.n))
        ).float()


class LatentFactor(ValuationModel):
    """Correlated valuations: vᵢ = ReLU(wᵢ·z + εᵢ)."""
    def __init__(self, n_bidders: int, sigma_eps: float = 0.05):
        super().__init__(n_bidders)
        self.w = torch.randn(n_bidders)
        self.sigma = sigma_eps

    def sample(self, n_samples: int) -> torch.Tensor:
        z   = torch.randn(n_samples, 1)
        eps = self.sigma * torch.randn(n_samples, self.n)
        return torch.relu(z @ self.w.unsqueeze(0) + eps)


class HeteroTypes(ValuationModel):
    """Mixture of three bidder-types."""
    def sample(self, n_samples: int) -> torch.Tensor:
        out = torch.empty(n_samples, self.n)
        for i in range(self.n):
            t = torch.randint(0, 3, (n_samples,))
            a, b, c = (t == k for k in range(3))
            out[a, i] = 0.5 + 0.5*torch.rand(a.sum())          # type-A
            out[b, i] = 0.5*torch.rand(b.sum())               # type-B
            out[c, i] = torch.from_numpy(
                np.random.exponential(scale=1.0, size=c.sum())
            ).float()                                         # type-C
        return out


# registry so CLI can pick a model by name
VAL_MODELS: Dict[str, Callable[[int], ValuationModel]] = {
    "uniform": IIDUniform,
    "exp":     IIDExponential,
    "latent":  LatentFactor,
    "hetero":  HeteroTypes,
}
