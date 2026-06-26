"""
Stage 4 (baselines) -- evaluate the NON-RL defenders on CAGE 2.

Runs random + the two react heuristics over many episodes and reports mean reward
with a 95% CI. The BEST heuristic here is the bar RL must clear (Stage 4 RL eval /
Stage 5 stats). Needs NO trained model -> you can run this while RL trains.

    python scripts/stage4_baselines.py --red bline --steps 30 --episodes 100
"""
import argparse
import json
import os

from eval_utils import eval_policy_gym, eval_heuristic, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--red", default="bline", choices=["bline", "meander", "sleep"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Baselines vs {args.red}: {args.episodes} episodes x {args.steps} steps "
          f"(seed {args.seed})")
    print("-" * 70)

    defenders = [
        ("random",            lambda: eval_policy_gym(None, args.red, args.steps,
                                                      args.episodes, args.seed)),
        ("heuristic_restore", lambda: eval_heuristic("restore", args.red, args.steps,
                                                     args.episodes, args.seed)),
        ("heuristic_remove",  lambda: eval_heuristic("remove", args.red, args.steps,
                                                     args.episodes, args.seed)),
    ]

    results = {}
    for name, fn in defenders:
        s = summarize(fn())
        results[name] = s
        print(f"{name:18s} mean {s['mean']:8.2f}  ±{s['ci95']:5.2f} (95% CI)   "
              f"[std {s['std']:.2f}]")

    best = max(results, key=lambda k: results[k]["mean"])
    print("-" * 70)
    print(f"BEST baseline = {best} (mean {results[best]['mean']:.2f}) "
          f"<- this is the bar RL must beat")

    path = os.path.join(args.out, f"stage4_baselines_{args.red}_s{args.steps}.json")
    with open(path, "w") as f:
        json.dump({"config": vars(args), "results": results, "best": best}, f, indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
