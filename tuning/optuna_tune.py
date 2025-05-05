"""
One-shot or resumable Optuna sweep with **local logging** and a
**per-distribution warm-start cache**.

- Run folder:  `auction_mech/tuning/runs/<timestamp>/<val_model>/`
      ├─ study.sqlite3   - resumable Optuna DB
      ├─ trials.csv      - every trial (+ val_model column)
      └─ best.json       - best λ's + objective

- Warm-start file: `auction_mech/tuning/warmstart/<val_model>.pt`
  Phase-1 imitation is executed **once per distribution** and reused for all
  subsequent trials, greatly accelerating sweeps.
"""
from __future__ import annotations

import argparse, json, optuna, pandas as pd, torch, sys
from pathlib import Path
from copy import deepcopy

from ..config import TrainConfig
from ..trainer import Trainer
from .utils import new_run_dir        # timestamped root folder

# --------------------------------------------------------------------
#  Warm-start directory
# --------------------------------------------------------------------
WARM_DIR = Path(__file__).parent / "warmstart"
WARM_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------
#  Objective for Optuna
# --------------------------------------------------------------------
def objective(trial: optuna.Trial, base_cfg: TrainConfig) -> float:
    # clone cfg so we can mutate λs
    cfg = deepcopy(base_cfg)

    cfg.λ_regret = trial.suggest_float("λ_regret", 0.1, 5.0,  log=True)
    cfg.λ_mono   = trial.suggest_float("λ_mono",   0.1, 2.0,  log=True)

    trainer = Trainer(cfg)

    ckpt = WARM_DIR / f"{cfg.val_model}.pt"
    if ckpt.exists():
        trainer.net.load_state_dict(torch.load(ckpt, map_location=cfg.device))
    else:
        trainer.phase1()                  # run once for this distribution
        torch.save(trainer.net.state_dict(), ckpt)

    trainer.phase2()
    rev, reg = trainer.evaluate()
    return rev - 5 * reg                  # maximise revenue – 5×regret


# --------------------------------------------------------------------
#  CLI entry-point
# --------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val_model", choices=["uniform", "exp", "latent", "hetero"],
                   default="uniform")
    p.add_argument("--trials", type=int, default=30,
                   help="number of Optuna trials to run / add")
    p.add_argument("--study_path", type=str, default=None,
                   help="resume an existing study.sqlite3 file")
    args = p.parse_args()

    # ---------- study storage ----------
    if args.study_path:                           # resume
        study_path = Path(args.study_path).expanduser()
        if not study_path.exists():
            raise FileNotFoundError(study_path)
        run_dir = study_path.parent
        storage_url = f"sqlite:///{study_path}"
        study = optuna.load_study(study_name="optuna", storage=storage_url)
        print(f"Resuming study at {study_path}")
    else:                                         # fresh run
        root_dir = new_run_dir()                  # e.g. runs/20250507-0912
        run_dir  = root_dir / args.val_model      # runs/…/uniform/
        run_dir.mkdir(parents=True, exist_ok=True)

        storage_url = f"sqlite:///{run_dir/'study.sqlite3'}"
        study = optuna.create_study(study_name="optuna",
                                    storage=storage_url,
                                    direction="maximize")
        print("Logging Optuna run to", run_dir)

    # ---------- optimise ----------
    base_cfg = TrainConfig(val_model=args.val_model)
    study.optimize(lambda t: objective(t, base_cfg), n_trials=args.trials)

    # ---------- save artefacts ----------
    df = study.trials_dataframe()
    df["val_model"] = args.val_model

    (run_dir / "trials.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    (run_dir / "best.json").write_text(
        json.dumps({"val_model": args.val_model,
                    "params": study.best_params,
                    "objective": study.best_value}, indent=2),
        encoding="utf-8"
    )

    print("Best params:", study.best_params,
          "\nBest objective:", study.best_value,
          "\nArtefacts saved to", run_dir)


if __name__ == "__main__":
    # Ensure same interpreter is used when launched via subprocess (auto_grid).
    sys.exit(main())