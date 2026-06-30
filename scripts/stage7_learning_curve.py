"""
Stage 7 -- reconstruct LEARNING CURVES from saved checkpoints.

Addresses the reviewer concern "two points is not a learning curve" WITHOUT any new
training. We did not enable SB3 TensorBoard/Monitor logging, but we checkpointed every
~20k steps to Drive. This script loads each checkpoint along training, evaluates it over
N episodes, and records (timestep, mean reward). Across seeds this yields a real learning
curve with a shaded ±1 SD band.

Run on Colab (Drive mounted), once per condition:
    python scripts/stage7_learning_curve.py --algo ppo --red bline   --seeds 5 --episodes 30
    python scripts/stage7_learning_curve.py --algo dqn --red bline   --seeds 5 --episodes 30
    python scripts/stage7_learning_curve.py --algo a2c --red bline   --seeds 5 --episodes 30
    python scripts/stage7_learning_curve.py --algo ppo --red meander --seeds 5 --episodes 30
"""
import argparse
import csv
import glob
import os

import numpy as np

from cyborg_gym import make_cyborg_env          # noqa: F401  (sets up paths/shim)
from checkpoint_utils import _ckpt_step
from eval_utils import eval_policy_gym, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ppo", choices=["ppo", "a2c", "dqn"])
    ap.add_argument("--red", default="bline", choices=["bline", "meander", "sleep"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=30,
                    help="eval episodes per checkpoint (lower = faster curve)")
    ap.add_argument("--drive_dir",
                    default="/content/drive/MyDrive/autonomous-cyber-defense-rl/checkpoints")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    from stable_baselines3 import PPO, A2C, DQN
    ALGO = {"ppo": PPO, "a2c": A2C, "dqn": DQN}[args.algo]

    rows = []
    for s in range(args.seeds):
        run_dir = os.path.join(
            args.drive_dir, f"{args.algo}_{args.red}_s{args.steps}_seed{s}")
        ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt_*.zip")), key=_ckpt_step)
        if not ckpts:
            print(f"seed {s}: no checkpoints in {run_dir} -- skip")
            continue
        for c in ckpts:
            step = _ckpt_step(c)
            model = ALGO.load(c)
            m = summarize(eval_policy_gym(model, args.red, args.steps,
                                          args.episodes, seed=s))["mean"]
            rows.append({"algo": args.algo, "red": args.red, "seed": s,
                         "timestep": step, "mean_reward": round(m, 3)})
            print(f"{args.algo} {args.red} seed{s} step {step:>7}: {m:8.2f}")

    if not rows:
        raise SystemExit("No checkpoints found. Mount Drive and check --drive_dir.")

    # --- save CSV (local + Drive) ---
    name = f"learning_curve_{args.algo}_{args.red}_s{args.steps}.csv"
    csv_path = os.path.join(args.out, name)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["algo", "red", "seed", "timestep", "mean_reward"])
        w.writeheader()
        w.writerows(rows)
    print("saved", csv_path)
    drive_results = os.path.join(os.path.dirname(args.drive_dir), "results")
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(drive_results, exist_ok=True)
        import shutil
        shutil.copy(csv_path, os.path.join(drive_results, name))
        print("saved", os.path.join(drive_results, name), "(persistent)")

    # --- figure: faint per-seed lines + mean +/- 1 SD band (bucketed to nearest 10k) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    seeds = sorted({r["seed"] for r in rows})
    for s in seeds:
        pts = sorted((r["timestep"], r["mean_reward"]) for r in rows if r["seed"] == s)
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color="#9ecae1", lw=1, alpha=0.7)
    buckets = {}
    for r in rows:
        b = int(round(r["timestep"] / 10000) * 10000)
        buckets.setdefault(b, []).append(r["mean_reward"])
    bx = sorted(buckets)
    bm = np.array([np.mean(buckets[b]) for b in bx])
    bsd = np.array([np.std(buckets[b]) for b in bx])
    ax.plot(bx, bm, color="#08519c", lw=2.3, label=f"{args.algo.upper()} mean")
    ax.fill_between(bx, bm - bsd, bm + bsd, color="#6baed6", alpha=0.3, label="±1 SD")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean episode reward (evaluation)")
    ax.set_title(f"Learning curve: {args.algo.upper()} vs {args.red} on CAGE 2\n"
                 f"{len(seeds)} seeds, {args.episodes} eval episodes per checkpoint")
    ax.legend()
    fig.tight_layout()
    fpath = os.path.join("figures", f"learning_curve_{args.algo}_{args.red}.png")
    fig.savefig(fpath, dpi=150)
    print("saved", fpath)


if __name__ == "__main__":
    main()
