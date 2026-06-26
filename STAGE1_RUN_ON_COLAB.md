# Stage 1 — run this on Google Colab, then paste me the output

Goal: install CAGE Challenge 2 and prove it runs by playing **one episode** with a
**random** defender. No GPU needed. This is just a sanity check.

> Why Colab and not your Mac: CAGE 2 needs old `gym` + Python ≤ 3.10. Your Mac has
> Python 3.14, which can't build it. Colab is Linux and works.

---

## Cell 1 — install (run it, then RESTART the runtime when it finishes)

```python
import os
# clone the official benchmark (skip if it is already there)
if not os.path.isdir("cage-challenge-2"):
    !git clone https://github.com/cage-challenge/cage-challenge-2.git

# Install ONLY the libraries CybORG needs. We do NOT pip-install CybORG itself:
# it is pure-Python and its setup.py breaks on modern pip, so Cell 2 just adds it
# to the Python path instead.
#   * gym 0.23.1 = old 4-tuple API (CybORG needs <0.26) and installs cleanly
#     (gym 0.21.0 fails to build).
#   * numpy 1.26.4 has a Colab wheel; CybORG uses no removed numpy aliases.
%pip install -q "gym==0.23.1" "numpy==1.26.4" paramiko pyyaml prettytable docutils

print("install finished — now click Runtime ▸ Restart runtime, then run Cell 2")
```

**After Cell 1, do `Runtime ▸ Restart runtime`** (the numpy change needs a
restart). The cloned repo and installed packages survive the restart — go
straight to Cell 2. *(The `numpy>=2` lines pip prints are harmless warnings from
Colab's preinstalled opencv/jax, which CAGE 2 does not use.)*

---

## Cell 2 — verify (the actual Stage 1 check)

```python
import sys, inspect, random
sys.path.insert(0, "/content/cage-challenge-2/CybORG")   # use CybORG without installing it
import numpy as np
from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent
from CybORG.Agents.Wrappers import ChallengeWrapper

SEED, STEPS = 0, 30
random.seed(SEED); np.random.seed(SEED)

red_cls = B_lineAgent          # the scripted attacker we load
path = inspect.getfile(CybORG)[:-10] + "/Shared/Scenarios/Scenario2.yaml"

print("CybORG version :", CYBORG_VERSION)
print("Scenario       : Scenario2 (CAGE Challenge 2)")
print("Red (attacker) :", red_cls.__name__, " <-- scripted adversary LOADED")
print("Blue (defender): RANDOM policy (sanity check only)")
print("-"*60)

cyborg = CybORG(path, "sim", agents={"Red": red_cls})
env = ChallengeWrapper(env=cyborg, agent_name="Blue")
env.action_space.seed(SEED)

obs = env.reset()
print("reset OK  -> obs length      =", len(obs))
print("action OK -> blue action size =", env.action_space.n)
print("-"*60)

total = 0.0
for _ in range(STEPS):
    action = env.action_space.sample()        # random defender
    obs, reward, done, info = env.step(action)
    total += reward

print("Episode finished WITHOUT errors.")
print(f"TOTAL REWARD (random vs {red_cls.__name__}) = {total:.2f}")
print("(reward <= 0 = accumulated damage; random should be poor — expected.)")
```

---

## What to send me back

Copy **all** the printed output of **both** cells — especially:
- the `CybORG version` line, the `action size`, and the `TOTAL REWARD` line, **or**
- the **full red error message** if anything fails (install errors are common and
  expected with CybORG — paste it and I'll fix the exact version conflict).

Then we **stop** and I wait for your OK before Stage 2 (Drive checkpointing).
