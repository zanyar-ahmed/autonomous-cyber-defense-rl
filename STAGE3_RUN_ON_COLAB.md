# Stage 3 — train RL defenders on CybORG (run on Colab)

We (1) verify the CybORG↔gymnasium bridge with a quick smoke test, then (2) train
the RL challengers (PPO / A2C / DQN) on CAGE 2, checkpointing to Google Drive.

> RL agents are **challengers**. The tuned heuristic is the primary baseline and is
> compared in Stage 4 — a single training reward here is **not** "the result".

---

## Cell 1 — install everything (fresh runtime is fine)

```python
import os
if not os.path.isdir("autonomous-cyber-defense-rl"):
    !git clone https://github.com/zanyar-ahmed/autonomous-cyber-defense-rl.git
if not os.path.isdir("cage-challenge-2"):
    !git clone https://github.com/cage-challenge/cage-challenge-2.git
%pip install -q "gym==0.23.1" "numpy==1.26.4" paramiko pyyaml prettytable docutils
%pip install -q "stable-baselines3==2.3.2"
print("installed — now Runtime ▸ Restart runtime, then run Cell 2")
```
**Then `Runtime ▸ Restart runtime`.**

## Cell 2 — SMOKE TEST (verify the bridge before any long training)

```python
%cd /content/autonomous-cyber-defense-rl
!python scripts/stage3_smoke_test.py --red bline --steps 30
```
Expected: `env_checker: PASSED`, a random-episode reward, `PPO mini-train: OK`, and
`SMOKE TEST PASSED`. **Paste me this output before training.** If it errors, paste the
error — we fix the bridge first.

## Cell 3 — mount Drive (for checkpoints)

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 4 — train the three RL agents (each resumes if Colab drops)

```python
%cd /content/autonomous-cyber-defense-rl
!python scripts/stage3_train.py --algo ppo --red bline --steps 30 --total 200000
!python scripts/stage3_train.py --algo dqn --red bline --steps 30 --total 200000
!python scripts/stage3_train.py --algo a2c --red bline --steps 30 --total 200000
```
Each prints `saved ckpt_...` lines into
`MyDrive/autonomous-cyber-defense-rl/checkpoints/<algo>_bline_s30_seed0/`.
If the session disconnects, **re-run the same line** — it resumes from the last
checkpoint (that's Stage 2 doing its job).

> Tip: 200k steps × 3 algos can take a while on CPU. You can start with `--total
> 50000` just to confirm the whole loop end-to-end, then scale up.

---

## What to send me back
1. The **smoke-test output** (Cell 2) — most important.
2. Once training runs, the `saved ckpt_...` lines so I can confirm checkpoints land
   on Drive.

Then I stop for your OK before Stage 4 (head-to-head evaluation vs the tuned
heuristic baseline, with the real metrics).
