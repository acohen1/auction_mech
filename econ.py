import torch

class EconLoss:
    """Static helpers for revenue & incentive constraints."""
    @staticmethod
    def revenue(pay):                       # maximise
        return pay.sum(dim=1).mean()

    @staticmethod
    def monotonicity(bids, alloc):          # penalty
        sbids, idx = bids.sort(dim=1, descending=True)
        salloc = alloc.gather(1, idx)
        return torch.relu(salloc[:, :-1] - salloc[:, 1:]).mean()

    @staticmethod
    def regret(model, bids, grid=11):       # coarse ex-post regret
        B, n = bids.shape
        a0, p0 = model(bids)
        truthful = (a0 * (bids - p0)).sum(dim=1)
        mis_vals = torch.linspace(0, 1, grid, device=bids.device)
        worst = torch.zeros(B, device=bids.device)
        for i in range(n):
            for v in mis_vals:
                mis = bids.clone(); mis[:, i] = v
                a2, p2 = model(mis)
                util2 = (a2 * (bids - p2)).sum(dim=1)
                worst = torch.maximum(worst, util2 - truthful)
        return worst.mean()
    
    @staticmethod
    def efficiency(bids, alloc):
        winner_pred = alloc.argmax(dim=1)
        winner_true = bids.argmax(dim=1)
        return (winner_pred == winner_true).float().mean()
    
    
