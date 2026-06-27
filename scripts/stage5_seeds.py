"""
Stage 5 -- multi-seed confirmation: is the RL win over the heuristic REAL?

For one (algo, attacker): train the RL agent on N seeds, evaluate each, and test the
per-seed means against the tuned heuristic baseline (t-test + Wilcoxon + Cohen's d).
Training is resumable per seed, so re-running after a Colab drop continues safely.

    python scripts/stage5_seeds.py --algo ppo --red bline --steps 30 \
        --total 50000 --seeds 5 --episodes 100

Run it again with --algo dqn / --algo a2c to check the win is not PPO-specific, and
with --red meander for a harder attacker.
"""
import argparse
import json
import os

from cyborg_gym import make_cyborg_env
from checkpoint_utils import resumable_train, find_latest_checkpoint
from eval_utils import eval_policy_gym, eval_heuristic, summarize
from stats_utils import compare_to_constant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ppo", choices=["ppo", "a2c", "dqn"])
    ap.add_argument("--red", default="bline", choices=["bline", "meander", "sleep"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--total", type=int, default=50000, help="train steps per seed")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=100, help="eval episodes per seed")
    ap.add_argument("--baseline", default="restore", choices=["restore", "remove"])
    ap.add_argument("--save_every", type=int, default=20000)
    ap.add_argument("--drive_dir",
                    default="/content/drive/MyDrive/autonomous-cyber-defense-rl/checkpoints")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not os.path.isdir("/content/drive/MyDrive"):
        raise SystemExit("Google Drive not mounted. Mount it first.")

    from stable_baselines3 import PPO, A2C, DQN
    ALGO = {"ppo": PPO, "a2c": A2C, "dqn": DQN}

    print(f"STAGE 5: {args.algo.upper()} vs heuristic_{args.baseline} on {args.red} | "
          f"{args.seeds} seeds x {args.total} steps | {args.episodes} eval episodes")
    print("=" * 72)

    # the bar (heuristic baseline), evaluated once
    base = summarize(eval_heuristic(args.baseline, args.red, args.steps,
                                    args.episodes, seed=0))
    print(f"baseline heuristic_{args.baseline}: mean {base['mean']:.2f} "
          f"± {base['ci95']:.2f}\n")

    per_seed_means = []
    for s in range(args.seeds):
        run_dir = os.path.join(
            args.drive_dir, f"{args.algo}_{args.red}_s{args.steps}_seed{s}")
        print(f"--- seed {s}: train ---")
        resumable_train(make_cyborg_env(red=args.red, max_steps=args.steps),
                        total_timesteps=args.total, run_dir=run_dir,
                        algo=args.algo, save_every=args.save_every, seed=s)
        latest, step = find_latest_checkpoint(run_dir)
        model = ALGO[args.algo].load(latest)
        m = summarize(eval_policy_gym(model, args.red, args.steps,
                                      args.episodes, seed=s))["mean"]
        per_seed_means.append(m)
        print(f"--- seed {s}: RL mean {m:.2f}  (ckpt {step})\n")

    res = compare_to_constant(per_seed_means, base["mean"], alt="greater")

    print("=" * 72)
    print(f"RL {args.algo.upper()} mean over {res['n_seeds']} seeds: "
          f"{res['rl_mean']:.2f}  95% CI [{res['rl_ci_lo']:.2f}, {res['rl_ci_hi']:.2f}]")
    print(f"heuristic baseline             : {res['baseline']:.2f}")
    print(f"Delta (RL - heuristic)         : {res['delta']:+.2f}")
    print(f"wins                           : {res['wins']}/{res['n_seeds']} seeds")
    print(f"t-test p (RL > baseline)       : {res['t_p']:.4f}")
    print(f"Wilcoxon p                     : {res['wilcoxon_p']:.4f}")
    print(f"Cohen's d                      : {res['cohen_d']:.2f}")
    verdict = ("REAL WIN (significant)" if res["delta"] > 0 and res["t_p"] < 0.05
               else "NOT significant -- needs more seeds/steps or RL does not win here")
    print(f"VERDICT                        : {verdict}")

    out = {"config": vars(args), "baseline_full": base, "stats": res}
    fname = f"stage5_{args.algo}_{args.red}_s{args.steps}_t{args.total}.json"
    path = os.path.join(args.out, fname)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {path}")

    # also persist a copy to Drive so results survive Colab session resets
    drive_results = os.path.join(os.path.dirname(args.drive_dir), "results")
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(drive_results, exist_ok=True)
        with open(os.path.join(drive_results, fname), "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {os.path.join(drive_results, fname)} (persistent)")


if __name__ == "__main__":
    main()
