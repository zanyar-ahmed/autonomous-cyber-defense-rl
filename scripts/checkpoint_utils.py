"""
Reusable Google-Drive checkpointing for RESUMABLE Stable-Baselines3 training.

Why this exists (project brief, Stage 2): Colab caps long runs and drops idle
sessions, so training must save to Google Drive and AUTOMATICALLY resume from the
latest checkpoint on the next run -- never restart from zero.

Used by:
  * scripts/stage2_checkpoint_demo.py  (Stage 2 -- verify the mechanism on CartPole)
  * the CybORG agents in Stage 3        (same function, real environment)

Resume strategy (standard SB3 pattern):
  * After every `save_every` steps we write  ckpt_<numtimesteps>.zip  to the Drive
    run directory, keeping ALL numbered checkpoints. Numbered (not overwritten)
    files mean a disconnect mid-save can at worst corrupt the newest file -- the
    previous good checkpoint still loads.
  * On start we glob ckpt_*.zip, pick the highest step, `PPO.load` it, and continue
    with reset_num_timesteps=False so the step counter keeps going.
  * Note: SB3 restores policy weights + num_timesteps (not the rollout buffer), the
    documented way to resume. Good enough; disclosed as a limitation.
"""
import glob
import json
import os
import time


def _ckpt_step(path):
    """ckpt_000010240.zip -> 10240"""
    base = os.path.basename(path)
    return int(base[len("ckpt_"):-len(".zip")])


def find_latest_checkpoint(run_dir):
    """Return (path, step) of the newest checkpoint in run_dir, or (None, 0)."""
    files = glob.glob(os.path.join(run_dir, "ckpt_*.zip"))
    if not files:
        return None, 0
    latest = max(files, key=_ckpt_step)
    return latest, _ckpt_step(latest)


def _save_meta(run_dir, **kw):
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(kw, f, indent=2)


def resumable_ppo_train(make_env, total_timesteps, run_dir,
                        save_every=5000, seed=0, ppo_kwargs=None, verbose=0):
    """
    Train a PPO agent to `total_timesteps`, checkpointing to `run_dir` (a Google
    Drive path) every `save_every` steps, resuming automatically if checkpoints
    already exist there.

    make_env : zero-arg callable returning a fresh (gymnasium) environment.
    Returns the trained SB3 model.
    """
    from stable_baselines3 import PPO

    os.makedirs(run_dir, exist_ok=True)
    env = make_env()

    latest, done_steps = find_latest_checkpoint(run_dir)
    if latest is None:
        print(f"[ckpt] no checkpoint in {run_dir}\n[ckpt] -> starting FRESH "
              f"(target {total_timesteps} steps, seed {seed})")
        model = PPO("MlpPolicy", env, seed=seed, verbose=verbose, **(ppo_kwargs or {}))
        first_call = True
    else:
        print(f"[ckpt] found {os.path.basename(latest)}\n[ckpt] -> RESUMING from "
              f"{done_steps}/{total_timesteps} steps")
        model = PPO.load(latest, env=env)
        first_call = False

    if model.num_timesteps >= total_timesteps:
        print(f"[ckpt] already complete at {model.num_timesteps} steps -- nothing to do.")
        return model

    while model.num_timesteps < total_timesteps:
        chunk = min(save_every, total_timesteps - model.num_timesteps)
        model.learn(chunk, reset_num_timesteps=first_call, progress_bar=False)
        first_call = False
        ckpt = os.path.join(run_dir, f"ckpt_{model.num_timesteps:09d}.zip")
        model.save(ckpt)
        _save_meta(run_dir, total_timesteps=total_timesteps, seed=seed,
                   last_step=model.num_timesteps,
                   updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"[ckpt] saved {os.path.basename(ckpt)}  "
              f"({model.num_timesteps}/{total_timesteps})")

    print(f"[ckpt] training COMPLETE at {model.num_timesteps} steps.")
    return model
