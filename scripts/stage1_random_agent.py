"""
Stage 1 -- benchmark sanity check for CAGE Challenge 2 (CybORG v2.1).

PURPOSE (per project brief, Stage 1):
  * Load the CAGE 2 environment (Scenario2).
  * State explicitly which scripted RED (attacker) agent is loaded.
  * Run a RANDOM blue (defender) agent for one complete episode.
  * Print the total reward and confirm the environment steps without errors.

This produces NO research result. It only verifies the benchmark is installed
and runnable. We build incrementally: verify this, then STOP for approval.

Run (after installing CAGE 2 -- see ../README.md):
    python scripts/stage1_random_agent.py --red bline --steps 30 --seed 0
"""
import argparse
import inspect
import random

import numpy as np

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper

# The scripted adversaries shipped with CAGE 2. "sleep" does nothing (no attack);
# "bline" goes straight for the goal; "meander" explores the network first.
RED_AGENTS = {
    "bline": B_lineAgent,
    "meander": RedMeanderAgent,
    "sleep": SleepAgent,
}


def scenario2_path():
    """Locate the Scenario2.yaml that ships inside the installed CybORG package."""
    path = inspect.getfile(CybORG)            # .../CybORG/CybORG.py
    return path[:-10] + "/Shared/Scenarios/Scenario2.yaml"


def main():
    ap = argparse.ArgumentParser(description="CAGE 2 Stage-1 sanity check")
    ap.add_argument("--red", default="bline", choices=list(RED_AGENTS),
                    help="scripted attacker to load (default: bline)")
    ap.add_argument("--steps", type=int, default=30,
                    help="episode length in steps (CAGE 2 uses 30/50/100)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    red_cls = RED_AGENTS[args.red]

    print(f"CybORG version : {CYBORG_VERSION}")
    print(f"Scenario       : Scenario2  (CAGE Challenge 2)")
    print(f"Red  (attacker): {red_cls.__name__}   <-- scripted adversary LOADED")
    print(f"Blue (defender): RANDOM policy  (sanity check only -- not a result)")
    print(f"Episode length : {args.steps} steps  |  seed {args.seed}")
    print("-" * 64)

    cyborg = CybORG(scenario2_path(), "sim", agents={"Red": red_cls})
    env = ChallengeWrapper(env=cyborg, agent_name="Blue")
    env.action_space.seed(args.seed)          # make the random defender reproducible

    obs = env.reset()
    print(f"reset OK   -> observation vector length = {len(obs)}")
    print(f"action OK  -> blue action-space size    = {env.action_space.n}")
    print("-" * 64)

    total_reward = 0.0
    for _ in range(args.steps):
        action = env.action_space.sample()    # random defender action
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print("Episode finished WITHOUT errors.")
    print(f"TOTAL REWARD (random defender vs {red_cls.__name__}) = {total_reward:.2f}")
    print("(Reward is <= 0: it is accumulated damage; more negative = worse. "
          "A random defender should score poorly -- that is expected and correct.)")


if __name__ == "__main__":
    main()
