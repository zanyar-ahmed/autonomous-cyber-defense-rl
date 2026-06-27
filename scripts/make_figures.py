"""
Stage 6/7 -- build the paper table + figure automatically from saved Stage-5 JSONs.

Reads every stage5_*.json (from --results, default the persistent Drive results dir,
falling back to local results/), prints a summary table, writes tables/summary.csv,
and saves figures/stage5_summary.png (RL mean +/- 95% CI per algo x attacker, with
the heuristic baseline drawn as a marker). Every number comes from the saved runs.

    python scripts/make_figures.py
"""
import argparse
import csv
import glob
import json
import os


def load_runs(results_dirs):
    runs = []
    seen = set()
    for d in results_dirs:
        for path in sorted(glob.glob(os.path.join(d, "stage5_*.json"))):
            key = os.path.basename(path)
            if key in seen:
                continue
            seen.add(key)
            with open(path) as f:
                obj = json.load(f)
            cfg, st = obj["config"], obj["stats"]
            runs.append({
                "algo": cfg["algo"], "red": cfg["red"], "steps": cfg["steps"],
                "seeds": st["n_seeds"], "baseline": st["baseline"],
                "rl_mean": st["rl_mean"], "ci_lo": st["rl_ci_lo"],
                "ci_hi": st["rl_ci_hi"], "delta": st["delta"],
                "wins": st["wins"], "t_p": st["t_p"], "wilcoxon_p": st["wilcoxon_p"],
                "cohen_d": st["cohen_d"],
                "significant": (st["delta"] > 0 and st["t_p"] < 0.05),
            })
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=[
        "/content/drive/MyDrive/autonomous-cyber-defense-rl/results", "results"])
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()
    os.makedirs(args.tables, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)

    runs = load_runs(args.results)
    if not runs:
        raise SystemExit("No stage5_*.json found. Run Stage 5 first.")

    # ---- table ----
    cols = ["algo", "red", "seeds", "baseline", "rl_mean", "ci_lo", "ci_hi",
            "delta", "wins", "t_p", "wilcoxon_p", "cohen_d", "significant"]
    print(f"{'algo':5} {'attacker':8} {'RLmean':>8} {'base':>8} {'delta':>7} "
          f"{'wins':>5} {'t_p':>7} {'d':>6}  verdict")
    print("-" * 70)
    for r in sorted(runs, key=lambda x: (x["red"], x["algo"])):
        verdict = "WIN" if r["significant"] else "ns"
        print(f"{r['algo']:5} {r['red']:8} {r['rl_mean']:8.2f} {r['baseline']:8.2f} "
              f"{r['delta']:+7.2f} {r['wins']:>5} {r['t_p']:7.3f} {r['cohen_d']:6.2f}"
              f"  {verdict}")
    csv_path = os.path.join(args.tables, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in runs:
            w.writerow({k: r[k] for k in cols})
    print(f"\nsaved {csv_path}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_s = sorted(runs, key=lambda x: (x["red"], x["algo"]))
    labels = [f"{r['algo'].upper()}\n{r['red']}" for r in runs_s]
    means = [r["rl_mean"] for r in runs_s]
    lo = [r["rl_mean"] - r["ci_lo"] for r in runs_s]
    hi = [r["ci_hi"] - r["rl_mean"] for r in runs_s]
    colors = ["#2a9d8f" if r["significant"] else "#bbbbbb" for r in runs_s]

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(runs_s)), 4.2))
    x = range(len(runs_s))
    ax.bar(x, means, yerr=[lo, hi], capsize=5, color=colors)
    # draw each run's heuristic baseline as a red dash
    for i, r in enumerate(runs_s):
        ax.plot([i - 0.4, i + 0.4], [r["baseline"], r["baseline"]],
                color="crimson", lw=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean episode reward (higher = better)")
    ax.set_title("RL defender vs tuned heuristic (red dash) on CAGE 2\n"
                 "green = significant win, grey = not significant")
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    fig_path = os.path.join(args.figures, "stage5_summary.png")
    fig.savefig(fig_path, dpi=150)
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
