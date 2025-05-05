from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

from .config import TrainConfig
from .valuation import VAL_MODELS
from .mechanism import AuctionNet
from .econ import EconLoss


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        # -------- data --------
        gen_cls = VAL_MODELS[cfg.val_model]
        gen = gen_cls(cfg.n_bidders)
        self.train_vals = gen.sample(cfg.train_size)
        self.val_vals = gen.sample(cfg.val_size)

        self.train_loader = DataLoader(
            TensorDataset(self.train_vals), cfg.batch, shuffle=True)
        self.val_loader = DataLoader(
            TensorDataset(self.val_vals), cfg.batch, shuffle=False)

        # -------- model -------
        self.net = AuctionNet(cfg.n_bidders).to(cfg.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)

    # ---------- target mechanisms ----------
    def _vickrey(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        top2, idx2 = bids.topk(2, dim=1)
        alloc = torch.zeros_like(bids).scatter_(1, idx2[:, :1], 1.0)
        pay = torch.zeros_like(bids).scatter_(1, idx2[:, :1], top2[:, 1:2])
        return alloc, pay

    def _myerson_uniform(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        virt = 2 * bids - 1                              # φ(v)=2v−1
        idx = virt.argmax(1, keepdim=True)
        alloc = torch.zeros_like(bids).scatter_(1, idx, 1.0)

        reserve = 0.5
        second = virt.clone().scatter_(1, idx, -float("inf")).max(1, keepdim=True).values
        price = torch.maximum(torch.full_like(second, reserve), (second + 1) / 2)
        pay = torch.zeros_like(bids).scatter_(1, idx, price)
        return alloc, pay

    def _myerson_exp(self, bids: torch.Tensor, lam: float = 1.0):
        virt = bids - 1 / lam                            # φ(v)=v−1/λ
        idx = virt.argmax(1, keepdim=True)
        alloc = torch.zeros_like(bids).scatter_(1, idx, 1.0)

        reserve = 1 / lam
        second = virt.clone().scatter_(1, idx, -float("inf")).max(1, keepdim=True).values
        price = torch.maximum(torch.full_like(second, reserve), second + 1 / lam)
        pay = torch.zeros_like(bids).scatter_(1, idx, price)
        return alloc, pay

    def _vickrey_reserve(self, bids: torch.Tensor, r: float = 1.0):
        top2, idx2 = bids.topk(2, dim=1)
        winner = top2[:, 0:1] >= r
        alloc = torch.zeros_like(bids).scatter_(1, idx2[:, :1], winner.float())
        pay_val = torch.maximum(torch.full_like(top2[:, :1], r), top2[:, 1:2])
        pay = torch.zeros_like(bids).scatter_(1, idx2[:, :1], winner.float() * pay_val)
        return alloc, pay

    # ---------- phase 1 ----------
    def phase1(self):
        L = nn.MSELoss()

        if self.cfg.val_model == "uniform":
            target_fn, imit_name = self._myerson_uniform, "myerson"
        elif self.cfg.val_model == "exp":
            target_fn, imit_name = self._myerson_exp, "myerson-exp"
        else:                                           # latent or hetero
            target_fn, imit_name = self._vickrey, "vickrey"

        for ep in range(self.cfg.phase1_epochs):
            self.net.train()
            for (b,) in self.train_loader:
                b = b.to(self.cfg.device)
                a, p = self.net(b)
                tgt_a, tgt_p = target_fn(b)
                loss = L(a, tgt_a) + L(p, tgt_p)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
            print(f"[phase1-{imit_name}] epoch {ep+1}: loss={loss.item():.4f}")

    # ---------- phase 2 ----------
    def phase2(self):
        for ep in range(self.cfg.phase2_epochs):
            self.net.train()
            for (b,) in self.train_loader:
                b = b.to(self.cfg.device)
                a, p = self.net(b)
                loss = (-EconLoss.revenue(p)
                        + self.cfg.λ_regret * EconLoss.regret(self.net, b)
                        + self.cfg.λ_mono   * EconLoss.monotonicity(b, a))
                self.opt.zero_grad(); loss.backward(); self.opt.step()
            print(f"[phase2] epoch {ep+1}: loss={loss.item():.4f}")

    # ---------- evaluation ----------
    def evaluate(self):
        self.net.eval()
        rev, reg, eff = [], [], []
        vrev, vreg = [], []
        brevs, bregs = [], []

        with torch.no_grad():
            for (b,) in self.val_loader:
                b = b.to(self.cfg.device)

                # learned auction
                a, p = self.net(b)
                rev.append(EconLoss.revenue(p).cpu())
                reg.append(EconLoss.regret(self.net, b).cpu())
                eff.append(EconLoss.efficiency(b, a).cpu())

                # Vickrey baseline
                va, vp = self._vickrey(b)
                vrev.append(EconLoss.revenue(vp).cpu())
                vreg.append(EconLoss.regret(lambda x: self._vickrey(x), b).cpu())

                # best-known truthful baseline per distribution
                if self.cfg.val_model == "uniform":
                    ba, bp = self._myerson_uniform(b)
                    bname = "Myerson"
                elif self.cfg.val_model == "exp":
                    ba, bp = self._myerson_exp(b)
                    bname = "Myerson-Exp"
                else:                                   # latent or hetero
                    ba, bp = self._vickrey_reserve(b, r=1.0)
                    bname = "Vickrey+Reserve(r=1)"

                brevs.append(EconLoss.revenue(bp).cpu())
                bregs.append(EconLoss.regret(lambda x: (ba, bp), b).cpu())

        mean = lambda xs: float(torch.stack(xs).mean())

        print(f"Learned   revenue={mean(rev):.3f}  regret={mean(reg):.4f}  eff={mean(eff):.3%}")
        print(f"Vickrey   revenue={mean(vrev):.3f}  regret={mean(vreg):.4f}")
        print(f"{bname:<12}  revenue={mean(brevs):.3f}  regret={mean(bregs):.4f}")

        return mean(rev), mean(reg)