from typing import List, Tuple
import torch
import torch.nn as nn


class AuctionNet(nn.Module):
    """
    Feed-forward single-item auction.
    For each bid profile it outputs:
      - alloc in [0,1]^{batch x n_bidders},  rows sum to 1
      - pay in [0,bid]  for the allocated bidder, 0 for others
    """
    def __init__(self,
                 n_bidders: int,
                 hidden: List[int] = [64, 32, 16]):
        super().__init__()
        layers = []
        dim = n_bidders
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.ReLU()]
            dim = h
        self.shared = nn.Sequential(*layers)

        # separate heads
        self.alloc_head = nn.Linear(dim, n_bidders)   # unnormalised scores
        self.pay_lin    = nn.Linear(dim, n_bidders)   # raw fractions in |R

    def forward(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        bids:  shape (batch, n_bidders), values in [0, 1] (or any scale you train on)
        returns:
            alloc: (batch, n_bidders) softmax over bidders
            pay:   (batch, n_bidders) payment vector, only winner pays
        """
        z = self.shared(bids)

        # allocation probabilities
        alloc = torch.softmax(self.alloc_head(z), dim=1)

        # fraction (0,1); multiply by own bid and alloc mask enforces IR
        frac = torch.sigmoid(self.pay_lin(z))
        pay  = frac * bids * alloc          # losers (alloc~0) pay 0

        return alloc, pay
