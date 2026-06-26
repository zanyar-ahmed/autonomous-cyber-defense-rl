# Stage 2 — Google Drive checkpoint + auto-resume (run on Colab)

Goal: prove that training **saves to Google Drive** and **automatically resumes**
after a Colab disconnect, instead of starting over. We test the mechanism on a
tiny fast env (CartPole) — **CartPole's reward is meaningless here**, we only care
that the 2nd run says `RESUMING from ...`.

This is independent of Stage 1 (no CybORG needed) — it can run in a fresh runtime.

---

## Cell 1 — get the code + install Stable-Baselines3

```python
import os
if not os.path.isdir("autonomous-cyber-defense-rl"):
    !git clone https://github.com/zanyar-ahmed/autonomous-cyber-defense-rl.git
%pip install -q "stable-baselines3==2.3.2"
print("ready")
```

## Cell 2 — mount your Google Drive (approve the popup)

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 3 — RUN 1 (fresh training → checkpoints to Drive)

```python
!python autonomous-cyber-defense-rl/scripts/stage2_checkpoint_demo.py --total 10000
```
You should see `starting FRESH` and several `saved ckpt_...` lines. The files land in
`MyDrive/autonomous-cyber-defense-rl/checkpoints/stage2_cartpole_demo/`.

## Cell 4 — RUN 2 (resume → must continue, not restart)

```python
!python autonomous-cyber-defense-rl/scripts/stage2_checkpoint_demo.py --total 20000
```
**This is the proof.** It should print `found ckpt_...` and **`RESUMING from
~10000/20000`**, then continue to 20000 — *not* start from 0.

### Real disconnect test (optional but convincing)
Run Cell 3 with `--total 60000`, and **press the stop button** after a couple of
`saved ckpt_` lines (simulating a Colab drop). Then run Cell 4 with `--total 60000`
— it resumes from the last saved step.

---

## What to send me back
Paste the output of **Cell 3 and Cell 4**. I'm looking for:
- Cell 3: `starting FRESH` + `saved ckpt_...` lines
- Cell 4: **`RESUMING from ...`** + it finishing at 20000

That confirms Stage 2. Then I stop for your OK before Stage 3 (the agents).
