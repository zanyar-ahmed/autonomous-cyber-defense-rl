# Stage 5 — full self-contained runbook (fresh Colab session)

Colab wipes everything when the session ends, so each new day you must re-install.
Run these cells **in order**. This trains PPO on several seeds and prints a VERDICT
on whether the win over the heuristic is statistically real.

---

## Cell 1 — install everything (run from a fresh runtime)

```python
import os
os.chdir("/content")
# 1) our code
if os.path.isdir("autonomous-cyber-defense-rl"):
    !cd /content/autonomous-cyber-defense-rl && git pull -q
else:
    !git clone https://github.com/zanyar-ahmed/autonomous-cyber-defense-rl.git
# 2) the CybORG / CAGE 2 benchmark
if not os.path.isdir("cage-challenge-2"):
    !git clone https://github.com/cage-challenge/cage-challenge-2.git
# 3) dependencies
!pip install -q "gym==0.23.1" "numpy==1.26.4" paramiko pyyaml prettytable docutils
!pip install -q "stable-baselines3==2.3.2"
!pip install -q "setuptools<81"
print("\nINSTALL DONE  ->  now click  Runtime ▸ Restart runtime,  then run Cell 2")
```

**Now click `Runtime ▸ Restart runtime`** (top menu). Do NOT re-run Cell 1.

## Cell 2 — connect Google Drive (approve the popup)

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 3 — quick check that everything imports (10 seconds)

```python
import os; os.chdir("/content/autonomous-cyber-defense-rl")
!python scripts/stage3_smoke_test.py --red bline --steps 30
```
You want to see `SMOKE TEST PASSED`. If yes, continue. If it errors, send me the error.

## Cell 4 — STAGE 5 (the real run)

```python
import os; os.chdir("/content/autonomous-cyber-defense-rl")
!python scripts/stage5_seeds.py --algo ppo --red bline --steps 30 --total 50000 --seeds 5 --episodes 100
```
- Trains PPO 5 times (different seeds), tests each vs the heuristic, prints a VERDICT.
- Takes ~1–1.5 h on CPU. **It saves to Drive after every seed**, so if Colab drops,
  just re-run this same cell — it skips finished seeds and continues.
- Want a faster first look? use `--seeds 3`.

---

## What to send me back
The final block from Cell 4: the `VERDICT`, `wins X/5`, the `t-test p`, `Wilcoxon p`,
and `Cohen's d`. That tells us if the RL win is statistically real.
