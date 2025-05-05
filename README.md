# Learning Revenue-Optimal Single-Item Auctions with Correlated Bidders

A lightweight sandbox for training revenue-optimal neural auctions under bidder correlation and distributional uncertainty.

📄 [Read the full project report (PDF)](./Final_Report.pdf)  
🔬 Implements latent-factor valuations, Optuna-based λ-tuning, and architectural incentive guarantees.
---

## Install

```bash
pip install torch numpy          # core
pip install optuna               # auto-tuning (pulls pandas & sqlalchemy)
```

*(No DB setup required: Optuna writes a self-contained `study.sqlite3` file.)*

---

## Example runs

```bash
# uniform(0,1) bidders, GPU if present
python -m auction_mech

# latent-factor bidders, force CPU
python -m auction_mech --val_model latent --device cpu
```

| flag          | default (*bold*)               |
|---------------|--------------------------------|
| `--val_model` | **uniform**, exp, latent, hetero |
| `--device`    | auto (cuda / cpu / mps)        |

Automatic Phase-1 imitation  

| `val_model` | warm-up rule |
|-------------|--------------|
| uniform     | Myerson (φ(v)=2v−1, r = 0.5) |
| exp         | Myerson-Exp (r = 1/λ) |
| latent      | Vickrey |
| hetero      | Vickrey |


### Output metrics

After training the CLI prints learned revenue, regret, and allocative
efficiency, plus Vickrey and distribution-specific baselines. Default λ-weights
for each distribution live in config.py; the CLI uses them whenever you skip Optuna.

---

## Hyper-parameter tuning (Optuna)

Phase-1 imitation is cached per distribution in `tuning/warmstart/`, so
each sweep runs it only once.

Objective = **revenue − 5 × regret** (modifiable in `optuna_tune.py`).

```bash
# 50-trial sweep – artefacts saved in tuning/runs/<timestamp>/
python -m auction_mech.tuning.optuna_tune --val_model latent --trials 50

# resume a latent model study and add 20 more trials to it
python -m auction_mech.tuning.optuna_tune \
       --study_path tuning/runs/20250503-1839/latent/study.sqlite3 \
       --trials 20
```

Optional: tune **all** distributions in one shot

```bash
python -m auction_mech.tuning.auto_grid          # default 20 trials each
```

> **Tip** `auto_grid.py` spawns each sweep with the same Python interpreter
> (`sys.executable`), so it will respect whatever virtual-env you have
> activated.

Run folder contents:

```
study.sqlite3   -> resumable Optuna storage
trials.csv      -> all trials (λ’s, revenue, regret, objective)
best.json       -> best λ’s + objective
```

---

## Layout

```
auction_mech/
  __main__.py        CLI launcher
  config.py          TrainConfig (λ’s chosen per val_model)
  valuation.py       valuation generators
  mechanism.py       AuctionNet (64-32-16 MLP, dual heads)
  econ.py            revenue / regret / IR / monotonicity
  trainer.py         two-phase loop + baselines
  tuning/
    utils.py         new_run_dir()
    optuna_tune.py   Optuna sweeps
    auto_grid.py     batch-run sweeps for all distributions
    warmstart/       cached Phase-1 checkpoints (.pt)
```

---

### Extend

* **New generator** -> subclass `ValuationModel` in `valuation.py`, register it.  
* **New constraint** -> add static fn in `econ.py`, include in loss in `trainer.py`.  
* **Alternative tuning** -> drop your own script into `tuning/`.
