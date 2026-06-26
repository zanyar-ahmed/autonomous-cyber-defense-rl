"""
Stage 3 -- train an RL defender (PPO / A2C / DQN) on CybORG / CAGE 2, with Google
Drive checkpointing + auto-resume (uses scripts/checkpoint_utils.py).

RL agents are CHALLENGERS. The tuned heuristic defender is the primary baseline and
is evaluated in Stage 4 -- do not read a single training reward here as "the result".

Run on Colab (after mounting Drive):
    python scripts/stage3_train.py --algo ppo  --red bline --steps 30 --total 200000
    python scripts/stage3_train.py --algo dqn  --red bline --steps 30 --total 200000
    python scripts/stage3_train.py --algo a2c  --red bline --steps 30 --total 200000

If Colab disconnects, just re-run the SAME command -- it resumes from the latest
Drive checkpoint.
"""
import argparse
import os

from cyborg_gym import make_cyborg_env
from checkpoint_utils import resumable_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ppo", choices=["ppo", "a2c", "dqn"])
    ap.add_argument("--red", default="bline", choices=["bline", "meander", "sleep"])
    ap.add_argument("--steps", type=int, default=30, help="episode horizon")
    ap.add_argument("--total", type=int, default=200000, help="total training timesteps")
    ap.add_argument("--save_every", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drive_dir",
                    default="/content/drive/MyDrive/autonomous-cyber-defense-rl/checkpoints")
    args = ap.parse_args()

    if not os.path.isdir("/content/drive/MyDrive"):
        raise SystemExit("Google Drive not mounted. Mount it first "
                         "(see STAGE2_RUN_ON_COLAB.md / STAGE3_RUN_ON_COLAB.md).")

    # one run dir per (algo, red, steps, seed) so checkpoints never collide
    run_name = f"{args.algo}_{args.red}_s{args.steps}_seed{args.seed}"
    run_dir = os.path.join(args.drive_dir, run_name)
    print(f"Algo={args.algo}  Red={args.red}  horizon={args.steps}  "
          f"total={args.total}  seed={args.seed}")
    print(f"Drive run dir: {run_dir}")
    print("-" * 64)

    resumable_train(
        make_env=make_cyborg_env(red=args.red, max_steps=args.steps),
        total_timesteps=args.total,
        run_dir=run_dir,
        algo=args.algo,
        save_every=args.save_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
