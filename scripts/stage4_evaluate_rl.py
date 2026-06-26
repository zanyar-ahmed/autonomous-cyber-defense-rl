"""
Stage 4 (RL vs heuristic) -- load trained RL checkpoints from Drive and compare
them head-to-head against the tuned heuristic baseline on CAGE 2.

The heuristic is the PRIMARY baseline. RL only "wins" if its mean reward is higher
AND the gap is not within noise (Stage 5 adds multi-seed CIs + significance tests).

    python scripts/stage4_evaluate_rl.py --red bline --steps 30 --episodes 100
"""
import argparse
import json
import os

from eval_utils import eval_policy_gym, eval_heuristic, summarize
from checkpoint_utils import find_latest_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["ppo", "dqn", "a2c"])
    ap.add_argument("--red", default="bline", choices=["bline", "meander", "sleep"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", default="restore", choices=["restore", "remove"])
    ap.add_argument("--drive_dir",
                    default="/content/drive/MyDrive/autonomous-cyber-defense-rl/checkpoints")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from stable_baselines3 import PPO, A2C, DQN
    ALGO = {"ppo": PPO, "a2c": A2C, "dqn": DQN}

    print(f"RL vs heuristic_{args.baseline} on {args.red}: "
          f"{args.episodes} episodes x {args.steps} steps")
    print("-" * 70)

    rows = {}
    base = summarize(eval_heuristic(args.baseline, args.red, args.steps,
                                    args.episodes, args.seed))
    rows[f"heuristic_{args.baseline}"] = base
    print(f"heuristic_{args.baseline:8s} (BASELINE) mean {base['mean']:8.2f} "
          f"±{base['ci95']:5.2f}")

    for algo in args.algos:
        run_dir = os.path.join(
            args.drive_dir, f"{algo}_{args.red}_s{args.steps}_seed{args.seed}")
        latest, step = find_latest_checkpoint(run_dir)
        if latest is None:
            print(f"{algo:18s} no checkpoint in {run_dir} -- train it first (skip)")
            continue
        model = ALGO[algo].load(latest)
        s = summarize(eval_policy_gym(model, args.red, args.steps,
                                      args.episodes, args.seed))
        rows[algo] = s
        delta = s["mean"] - base["mean"]
        verdict = "BEATS heuristic" if delta > 0 else "loses to heuristic"
        print(f"{algo:18s} mean {s['mean']:8.2f} ±{s['ci95']:5.2f}  "
              f"(ckpt {step})  Δ={delta:+.2f} -> {verdict}")

    path = os.path.join(
        args.out, f"stage4_rl_vs_heuristic_{args.red}_s{args.steps}.json")
    with open(path, "w") as f:
        json.dump({"config": vars(args), "results": rows}, f, indent=2)
    print("-" * 70)
    print(f"saved {path}")
    print("NOTE: a positive Δ is only a *candidate* win -- Stage 5 confirms it with "
          "multiple seeds, CIs and a significance test before we believe it.")


if __name__ == "__main__":
    main()
