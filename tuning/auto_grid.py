import subprocess, sys

dists  = ["uniform", "exp", "latent", "hetero"]
trials = 20

for d in dists:
    print(f"\n=== tuning {d} ===")
    subprocess.run([
        sys.executable,
        "-m", "auction_mech.tuning.optuna_tune",
        "--val_model", d,
        "--trials", str(trials)
    ], check=True)
