# Stage 4 — evaluate defenders head-to-head (run on Colab)

We measure each defender's mean reward over many episodes and ask the core question:
**does any RL agent beat the tuned heuristic?** The heuristic is the primary baseline.

> A positive RL gap here is only a *candidate* win — Stage 5 confirms it with multiple
> seeds + confidence intervals + a significance test before we believe it.

Always `git pull` first so you have the latest scripts:
```python
%cd /content/autonomous-cyber-defense-rl
!git pull -q
```

## 4a — Baselines (NO training needed → run this anytime)

```python
%cd /content/autonomous-cyber-defense-rl
!python scripts/stage4_baselines.py --red bline --steps 30 --episodes 100
```
Prints `random`, `heuristic_restore`, `heuristic_remove` mean ± 95% CI, and tells you
the **BEST baseline** — the bar RL must clear. Saved to `results/`.

## 4b — RL vs heuristic (after training in Stage 3)

Needs the trained checkpoints on Drive (mount Drive first), then:
```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/autonomous-cyber-defense-rl
!python scripts/stage4_evaluate_rl.py --red bline --steps 30 --episodes 100
```
For each of PPO/DQN/A2C it prints mean ± CI, the checkpoint step, and
`Δ vs heuristic` with `BEATS` / `loses`. Saved to `results/`.

---

## What to send me back
- 4a baseline table (especially the BEST baseline number).
- 4b RL-vs-heuristic table once training has produced checkpoints.

Then Stage 5: repeat across 10–15 seeds with full statistics (Wilcoxon, Cohen's d,
Holm) to decide whether any RL win is real.
