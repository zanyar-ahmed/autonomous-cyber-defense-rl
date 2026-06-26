# Paper 2 — Reinforcement Learning for Autonomous Cyber Defense

When (if ever) does Reinforcement Learning *measurably* beat a carefully tuned
heuristic defender for autonomous cyber **response**, as an adaptive attacker
changes the environment over time?

This is the complementary question to Paper 1, which showed RL does **not** beat
a tuned threshold for intrusion **detection** (a threshold-reducible problem).

## Benchmark: CAGE Challenge 2 (CybORG v2.1)

We use the official **CybORG** simulator with the **CAGE Challenge 2** benchmark
(Scenario2). A blue (defender) agent protects a network while a *scripted* red
(attacker) agent — **B-line**, **Meander**, or **Sleep** — tries to reach and
impact the operational server.

**Why CAGE 2 and not the newer CAGE 4?** CAGE 2 (2022) has the largest body of
*published baselines and champion agents*, which is exactly what a fair "RL vs
strong heuristic" comparison needs. Using an established public benchmark also
removes Paper 1's main caveat (results were in our own simulator).

## Research integrity rules (carried from Paper 1)

- Never fabricate, smooth, or estimate a number — every value comes from code that ran.
- RL is a **challenger**; the **tuned heuristic defender is the primary baseline**.
- Any claimed win needs multiple seeds + 95% CI + a significance test.
- Every table/figure must regenerate from saved `results/` logs.

## Status (incremental — one stage at a time)

| Stage | What | State |
|------|------|-------|
| 1 | Install CAGE 2; run a **random** defender for one episode; print reward | **ready to run on Colab** |
| 2 | Google Drive checkpointing / auto-resume after Colab disconnect | not started |
| 3 | Agents: tuned heuristic (primary), PPO, A2C/DQN, (champion if runnable) | not started |
| 4 | Evaluate vs progressively stronger attackers (B-line, Meander, …) | not started |
| 5 | 10–15 seeds; mean/SD/95% CI/Wilcoxon/Cohen's d/Holm | not started |
| 6 | Reproducibility: CSV+JSON+plots+config+seed+timestamp per run | not started |
| 7 | Analysis: learning curves, reward distributions, effect sizes | not started |

## Install (Colab — the supported environment)

CAGE 2 needs **Python 3.7–3.10** and an **old `gym`**, so it does **not** run on
this Mac's Python 3.14. Run it on Colab. Exact steps + the verification cell:
see **`STAGE1_RUN_ON_COLAB.md`**. Pinned versions are in `requirements.txt`.

## Layout

```
scripts/    experiment + verification scripts (each says what it produces)
configs/    experiment configs
results/    raw CSV/JSON logs (source of every number)
figures/    plots, regenerated from results/
tables/     paper tables, regenerated from results/
logs/       run logs
notebooks/  Colab notebooks
```
