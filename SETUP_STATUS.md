# CPSC 6430 T2 — Setup Status (Checkpoint 1, practical start)
Date: 2026-10-20. Full command-by-command evidence: `setup_log.md`. Smoke-test output: `smoke_test_output.txt`.

## TL;DR
The full LIBERO stack is **installed and verified working end-to-end**: a LIBERO-Goal environment was created, reset with a seed, stepped, and its success checker called. Setup remains "in progress" only for dataset download (Checkpoint 2 item).

## Installed OK (verified by import + smoke test)
| Component | Version | Notes |
|---|---|---|
| Python | 3.12.3 | venv at `~/workspace/cpsc6430_t2/venv` |
| PyTorch | 2.14.1+cpu | CPU wheel; `cuda.is_available()` = False (expected) |
| MuJoCo | 3.1.6 | Downgraded from 3.16.0 — see below |
| robosuite | 1.4.1 | Downgraded from 1.5.2 — see below |
| LIBERO | 0.1.0 (repo HEAD) | Cloned from github.com/Lifelong-Robot-Learning/LIBERO |
| numpy | 1.26.4 | Pinned back after a transitive upgrade to 2.5.4 broke the stack |
| gym | 0.25.2 | LIBERO's pinned version |
| bddl / hydra-core / easydict / einops / cloudpickle / pyyaml | 3.6.0 / 1.3.7 / 1.13 / 0.8.2 / 3.1.2 / 6.0.3 | LIBERO runtime deps (its setup.py declares none) |

## What failed and how it was resolved (details in setup_log.md)
1. **PyTorch first download timed out** (CDN hiccup) → retry with longer timeout succeeded.
2. **LIBERO's `pip install -e .` is silently broken** (empty editable finder mapping) → fixed with a `libero_dev.pth` path file in site-packages.
3. **LIBERO requirements.txt not usable verbatim** — 2022 pins (numpy==1.22.4, robosuite==1.4.0, transformers==4.21.1…) are incompatible with Python 3.12; installed compatible versions of the needed subset instead.
4. **robosuite 1.5.2 incompatible with LIBERO** (`single_arm_env` module removed) → downgraded to robosuite 1.4.1.
5. **No EGL/OSMesa on this headless VM** and robosuite macros force `MUJOCO_GL=egl` → worked around with `MUJOCO_GL=glx` (macros exemption) + **Xvfb** virtual display (found preinstalled). apt could not install libegl1/libosmesa6 — Ubuntu mirrors unreachable from this VM (apt-get update hung).
6. **MuJoCo 3.16 incompatible with robosuite 1.4.1's binding layer** (joint-type assert) → downgraded to mujoco 3.1.6.

## Smoke test results (`smoke_test.py`, run: `xvfb-run -a ./venv/bin/python smoke_test.py`)
- Suites: libero_10, libero_100, libero_90, **libero_goal**, libero_object, libero_spatial
- libero_goal: 10 tasks; task 0 = "open the middle drawer of the cabinet", task 1 = "put the bowl on the stove" (two different language goals available for the T2 plan)
- `OffScreenRenderEnv` created; `seed(42)` + `reset()` OK
- **Observation keys (25):** `robot0_proprio-state` (39,), `object-state` (56,), `robot0_eef_pos/quat`, `robot0_joint_pos/vel` (7,), `robot0_gripper_qpos/qvel` (2,), per-object pos/quat and eef-relative pos/quat; camera check also passed: `agentview_image` (128,128,3), `robot0_eye_in_hand_image` (128,128,3)
- **Action space: 7 components, each in [-1, 1]** (6-DoF end-effector delta + 1 gripper) — matches T2's "seven action components"
- Random step OK (reward 0.0, done=False); `env.check_success()` callable → False (expected)

## Next actions
1. Download LIBERO demonstration datasets (datasets dir currently absent) — needed before Checkpoint 2 behavior cloning.
2. Install training-time deps when needed (robomimic, transformers, wandb) at Python-3.12-compatible versions.
3. Checkpoint 2: complete reset/step/replay + saved-state restore checks; record two LIBERO-Goal tasks' demo collection plan (60–100 sim demos).
4. If a GPU machine becomes available, re-verify with `MUJOCO_GL=egl` and install libegl1/libosmesa6 there; the Xvfb+GLX recipe is the fallback for this VM.
