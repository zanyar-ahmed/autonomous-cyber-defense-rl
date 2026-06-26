"""
Gymnasium adapter for CybORG / CAGE Challenge 2 (Scenario2, Blue defender).

WHY: CybORG's ChallengeWrapper speaks the OLD gym API (gym.spaces, reset->obs,
4-tuple step). Stable-Baselines3 v2 requires the gymnasium API (gymnasium.spaces,
reset->(obs, info), step->(obs, reward, terminated, truncated, info)). This thin
Env translates between them. It does NOT modify any CybORG dynamics.

This is the honest "use gymnasium" piece (see Stage-1 notes): the RL agents really
do train through a gymnasium interface, while CybORG itself still runs on old gym
underneath -- we only adapt the boundary.

CAGE 2 episodes have a FIXED horizon and no goal/absorbing state, so episodes end
by TIME LIMIT -> truncated=True (terminated stays False). That is the correct
gymnasium semantics (bootstrapping should continue past a time-limit cutoff).
"""
import inspect
import os
import sys

# locate the cloned CybORG (pure-Python, used via sys.path -- see Stage 1)
for _p in (
    os.environ.get("CYBORG_PATH", ""),
    "cage-challenge-2/CybORG",
    "/content/cage-challenge-2/CybORG",
    os.path.expanduser("~/cage-challenge-2/CybORG"),
):
    if _p and os.path.isdir(os.path.join(_p, "CybORG")) and _p not in sys.path:
        sys.path.insert(0, _p)
        break

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from CybORG.Agents.Wrappers import ChallengeWrapper

RED_AGENTS = {"bline": B_lineAgent, "meander": RedMeanderAgent, "sleep": SleepAgent}


def scenario2_path():
    p = inspect.getfile(CybORG)
    return p[:-10] + "/Shared/Scenarios/Scenario2.yaml"


class CyborgBlueEnv(gym.Env):
    """CAGE 2 Scenario2 Blue defender as a gymnasium environment.

    Parameters
    ----------
    red : {"bline", "meander", "sleep"}   scripted attacker to face.
    max_steps : episode horizon (CAGE 2 uses 30/50/100).
    """
    metadata = {"render_modes": []}

    def __init__(self, red="bline", max_steps=100):
        super().__init__()
        if red not in RED_AGENTS:
            raise ValueError(f"red must be one of {list(RED_AGENTS)}, got {red!r}")
        self.red_name = red
        self.max_steps = int(max_steps)

        self._make_inner()
        # mirror CybORG's own spaces into gymnasium spaces
        self.action_space = spaces.Discrete(int(self._env.action_space.n))
        o = self._env.observation_space
        self.observation_space = spaces.Box(
            low=np.asarray(o.low, dtype=np.float32),
            high=np.asarray(o.high, dtype=np.float32),
            dtype=np.float32,
        )
        self._steps = 0

    def _make_inner(self):
        cyborg = CybORG(scenario2_path(), "sim",
                        agents={"Red": RED_AGENTS[self.red_name]})
        self._env = ChallengeWrapper(env=cyborg, agent_name="Blue")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            try:
                self._env.action_space.seed(seed)
            except Exception:
                pass
        out = self._env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        self._steps = 0
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        out = self._env.step(int(action))
        if len(out) == 5:                      # already gymnasium-style
            obs, reward, terminated, truncated, _ = out
        else:                                  # old gym 4-tuple (CybORG default)
            obs, reward, done, _ = out
            terminated, truncated = False, bool(done)
        self._steps += 1
        if self._steps >= self.max_steps:      # time-limit truncation
            truncated = True
        # keep info slim -> avoids SB3 VecEnv copying CybORG's heavy result object
        return (np.asarray(obs, dtype=np.float32), float(reward),
                bool(terminated), bool(truncated), {})

    @property
    def unwrapped_cyborg(self):
        """The underlying CybORG ChallengeWrapper (for Stage-4 metric queries)."""
        return self._env


def make_cyborg_env(red="bline", max_steps=100):
    """Zero-arg-friendly factory for SB3 / checkpoint_utils."""
    def _f():
        return CyborgBlueEnv(red=red, max_steps=max_steps)
    return _f
