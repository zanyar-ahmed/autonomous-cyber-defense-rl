"""
Stage 3 smoke test -- verify the CybORG <-> gymnasium bridge works with SB3.

Three quick checks (no Drive, no long training):
  1. SB3's env_checker validates CyborgBlueEnv (spaces, reset/step API).
  2. A random episode runs through the gymnasium interface and prints reward.
  3. PPO trains for a tiny number of steps WITHOUT error (proves the full RL path).

If this passes, Stage 3 real training (stage3_train.py) is safe to launch.

    python scripts/stage3_smoke_test.py --red bline --steps 30
"""
import argparse

from cyborg_gym import CyborgBlueEnv          # same scripts/ folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--red", default="bline", choices=["bline", "meander", "sleep"])
    ap.add_argument("--steps", type=int, default=30, help="episode horizon")
    args = ap.parse_args()

    print("=== 1) build env + SB3 env_checker ===")
    env = CyborgBlueEnv(red=args.red, max_steps=args.steps)
    print("observation_space:", env.observation_space)
    print("action_space     :", env.action_space)
    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)        # raises if the env is not SB3-compatible
    print("env_checker: PASSED")

    print("\n=== 2) one random episode through the gymnasium API ===")
    obs, info = env.reset(seed=0)
    total, t, term, trunc = 0.0, 0, False, False
    while not (term or trunc):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        t += 1
    print(f"random episode: {t} steps, total reward = {total:.2f} "
          f"(terminated={term}, truncated={trunc})")

    print("\n=== 3) PPO trains 1000 steps without error ===")
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", CyborgBlueEnv(red=args.red, max_steps=args.steps),
                seed=0, verbose=0, n_steps=256)
    model.learn(1000)
    print("PPO mini-train: OK")

    print("\nSMOKE TEST PASSED -- the bridge + SB3 training path work. "
          "Stage 3 real training is safe to run.")


if __name__ == "__main__":
    main()
