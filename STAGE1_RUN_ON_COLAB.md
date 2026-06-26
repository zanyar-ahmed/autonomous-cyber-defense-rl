# Stage 1 — run this on Google Colab, then paste me the output

Goal: install CAGE Challenge 2 and prove it runs by playing **one episode** with a
**random** defender. No GPU needed. This is just a sanity check.

> Why Colab and not your Mac: CAGE 2 needs old `gym` + Python ≤ 3.10. Your Mac has
> Python 3.14, which can't build it. Colab is Linux and works.

---

## Cell 1 — install (run it, then RESTART the runtime when it finishes)

```python
# clone the official benchmark
!git clone https://github.com/cage-challenge/cage-challenge-2.git

# build tools that let the old gym 0.21 install on modern pip
%pip install -q "setuptools==65.5.0" "wheel==0.38.4"

# install CybORG (CAGE 2) in editable mode
%pip install -q -e cage-challenge-2/CybORG

# pin the two version-sensitive libraries (this is the part that usually breaks)
%pip install -q "gym==0.21.0" "numpy==1.23.5"

print("install finished — now click Runtime ▸ Restart runtime, then run Cell 2")
```

**After Cell 1, do `Runtime ▸ Restart runtime`** (the numpy downgrade needs a
restart). Do **not** re-run Cell 1 after restarting — go straight to Cell 2.

---

## Cell 2 — verify (the actual Stage 1 check)

```python
import inspect, random
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
