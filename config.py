from dataclasses import dataclass, field
import torch

# ---- per-distribution λ defaults (replace with your Optuna winners) ----
DEFAULT_LAMBDAS = {
    "uniform": dict(λ_regret=2.5088595885789813, λ_mono=0.28165351037978287),
    "exp":     dict(λ_regret=2.9029276978558367, λ_mono=0.4687319992036483),
    "latent":  dict(λ_regret=1.8975934768008145, λ_mono=0.663845376780487),
    "hetero":  dict(λ_regret=4.397392970362066, λ_mono=0.22685112547224978),
}

@dataclass
class TrainConfig:
    # universal settings
    n_bidders: int   = 4
    train_size: int  = 50_000
    val_size: int    = 10_000
    batch: int       = 512
    lr: float        = 3e-4
    phase1_epochs: int = 5
    phase2_epochs: int = 20
    device: str      = "cuda" if torch.cuda.is_available() else "cpu"

    # experiment selector
    val_model: str   = "uniform"

    # λ-weights (filled automatically in __post_init__)
    λ_regret: float  = field(init=False)
    λ_mono: float    = field(init=False)

    # ---- derive λs from val_model ----
    def __post_init__(self):
        if self.val_model not in DEFAULT_LAMBDAS:
            raise ValueError(f"Unknown val_model: {self.val_model}")
        lambdas = DEFAULT_LAMBDAS[self.val_model]
        self.λ_regret = lambdas["λ_regret"]
        self.λ_mono   = lambdas["λ_mono"]
