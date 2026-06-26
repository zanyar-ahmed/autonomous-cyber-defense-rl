"""
Stage 2 -- verify Google Drive checkpoint + AUTO-RESUME (infrastructure test).

We prove the save -> disconnect -> resume cycle works end to end using a tiny,
fast stand-in environment (CartPole-v1), with the SAME code Stage 3 will use on
CybORG (scripts/checkpoint_utils.py:resumable_ppo_train).

This produces NO research result. CartPole's reward here is irrelevant -- the only
thing we verify is that re-running RESUMES from the Drive checkpoint instead of
restarting from zero.

Typical use on Colab (after mounting Drive -- see STAGE2_RUN_ON_COLAB.md):
    # Run 1 (fresh): trains to 10000 and checkpoints to Drive
    python scripts/stage2_checkpoint_demo.py --total 10000
    # Run 2 (resume): same Drive dir, larger target -> continues, not restarts
    python scripts/stage2_checkpoint_demo.py --total 20000
"""
import argparse
import os

import gymnasium as gym

from checkpoint_utils import resumable_ppo_train   # same scripts/ folder


def make_env():
    return gym.make("CartPole-v1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive_dir",
                    default="/content/drive/MyDrive/autonomous-cyber-defense-rl/checkpoints")
    ap.add_argument("--run_name", default="stage2_cartpole_demo")
    ap.add_argument("--total", type=int, default=20000)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isdir("/content/drive/MyDrive"):
        raise SystemExit(
            "Google Drive is not mounted. Run the mount cell first:\n"
            "    from google.colab import drive; drive.mount('/content/drive')\n"
            "(see STAGE2_RUN_ON_COLAB.md)")

    run_dir = os.path.join(args.drive_dir, args.run_name)
    print(f"Drive run dir : {run_dir}")
    print(f"Target steps  : {args.total}  (checkpoint every {args.save_every})")
    print("-" * 64)
    resumable_ppo_train(make_env, args.total, run_dir,
                        save_every=args.save_every, seed=args.seed)
    print("-" * 64)
    print("Stage 2 check: if you ran this twice, the 2nd run should have printed "
          "'RESUMING from ...' -- that confirms Drive checkpoint + resume works.")


if __name__ == "__main__":
    main()
