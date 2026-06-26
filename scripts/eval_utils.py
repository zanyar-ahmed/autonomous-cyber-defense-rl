"""
Evaluation utilities (Stage 4) -- run defenders on CAGE 2 and collect rewards.

Two reward-EQUIVALENT paths (both sum the same underlying CybORG reward, so RL and
heuristic numbers are directly comparable -- the fairness point):
  * eval_policy_gym -- RL agent OR random, through the gymnasium bridge (CyborgBlueEnv).
  * eval_heuristic  -- CybORG's react/restore heuristic, through the NATIVE CybORG
    API (the heuristic reads the raw dict observation, not the flattened vector).

Higher (less negative) reward = better defense.
"""
import random

import numpy as np

from cyborg_gym import CyborgBlueEnv, RED_AGENTS, scenario2_path


def summarize(rewards):
    r = np.asarray(rewards, dtype=float)
    n = len(r)
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    ci = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0   # normal-approx 95% CI
    return {"n": n, "mean": mean, "std": std, "ci95": ci,
            "min": float(r.min()), "max": float(r.max())}


def eval_policy_gym(model, red="bline", steps=30, n_episodes=100, seed=0):
    """Evaluate an SB3 model (or random if model is None) via the gymnasium bridge.
    Returns a list of per-episode total rewards."""
    random.seed(seed)
    np.random.seed(seed)
    env = CyborgBlueEnv(red=red, max_steps=steps)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            total += r
            done = term or trunc
        rewards.append(total)
    return rewards


def eval_heuristic(kind="restore", red="bline", steps=30, n_episodes=100, seed=0):
    """Evaluate CybORG's react heuristic natively. kind in {'restore','remove'}.
    Returns a list of per-episode total rewards."""
    from CybORG import CybORG
    from CybORG.Agents.SimpleAgents.BlueReactAgent import (
        BlueReactRestoreAgent, BlueReactRemoveAgent)

    AgentCls = {"restore": BlueReactRestoreAgent,
                "remove": BlueReactRemoveAgent}[kind]
    random.seed(seed)
    np.random.seed(seed)
    rewards = []
    for ep in range(n_episodes):
        cyborg = CybORG(scenario2_path(), "sim", agents={"Red": RED_AGENTS[red]})
        blue = AgentCls()
        res = cyborg.reset(agent="Blue")
        obs = res.observation
        acts = cyborg.get_action_space("Blue")
        total = 0.0
        for _ in range(steps):
            action = blue.get_action(obs, acts)
            res = cyborg.step(agent="Blue", action=action)
            obs = res.observation
            acts = cyborg.get_action_space("Blue")
            total += res.reward
        blue.end_episode()
        rewards.append(total)
    return rewards
