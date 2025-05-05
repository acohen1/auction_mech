"""
CLI launcher:  python -m auction_mech --val_model latent --device cpu
"""
import argparse, pprint
from dataclasses import asdict

from .config import TrainConfig
from .trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_model",
                        choices=["uniform", "exp", "latent", "hetero"],
                        default="uniform")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # build config
    cfg = TrainConfig(val_model=args.val_model)
    if args.device:
        cfg.device = args.device

    # figure out which imitation rule Trainer will use
    imit_name = "myerson" if args.val_model in {"uniform", "exp"} else "vickrey"

    print("Config")
    pprint.pp(asdict(cfg))

    trainer = Trainer(cfg)
    print(f"\nPhase 1: imitate {imit_name}")
    trainer.phase1()

    print("\nPhase 2: direct optimisation")
    trainer.phase2()

    print("\nEvaluation")
    trainer.evaluate()


if __name__ == "__main__":
    main()
