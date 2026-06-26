"""
Statistics for Stage 5 -- decide whether an RL win is REAL or noise.

We collect one mean-reward number PER TRAINING SEED for the RL agent, then test
whether that set of per-seed means is significantly better than the (strong) tuned
heuristic baseline. Same rigor as Paper 1: t-CI, one-sample t-test, Wilcoxon
signed-rank, Cohen's d, and an explicit win count.
"""
import numpy as np
from scipy import stats


def t_ci(x, conf=0.95):
    """Mean and (lo, hi) two-sided t confidence interval."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, m, m
    se = x.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf((1 + conf) / 2, n - 1)
    return m, m - h, m + h


def cohen_d_one_sample(x, mu):
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float((x.mean() - mu) / sd) if sd > 0 else float("inf")


def compare_to_constant(rl_means, baseline, alt="greater"):
    """Test whether per-seed RL means beat a baseline constant.

    Returns a dict with the RL mean+CI, the gap vs baseline, a one-sample t-test
    p-value, a Wilcoxon signed-rank p-value, Cohen's d, and how many seeds won.
    """
    x = np.asarray(rl_means, dtype=float)
    n = len(x)
    mean, lo, hi = t_ci(x)

    if n > 1:
        t_p = float(stats.ttest_1samp(x, baseline, alternative=alt).pvalue)
    else:
        t_p = float("nan")

    diffs = x - baseline
    try:
        w_p = float(stats.wilcoxon(diffs, alternative=alt).pvalue)
    except Exception:
        w_p = float("nan")

    return {
        "n_seeds": n,
        "rl_mean": mean,
        "rl_ci_lo": lo,
        "rl_ci_hi": hi,
        "baseline": float(baseline),
        "delta": mean - float(baseline),
        "t_p": t_p,
        "wilcoxon_p": w_p,
        "cohen_d": cohen_d_one_sample(x, baseline),
        "wins": int((x > baseline).sum()),
        "per_seed": [float(v) for v in x],
    }
