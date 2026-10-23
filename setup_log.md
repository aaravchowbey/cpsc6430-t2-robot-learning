# CPSC 6430 T2 — LIBERO stack setup log

## Environment — 2026-10-20 08:47 EDT
- Command: `python3 --version; uname -a; python3 -m venv venv; pip --version`
- Python: 3.12.3
- OS: Linux htch-runtime 7.0.0-39-generic x86_64 (Ubuntu-based VM)
- venv: `~/workspace/cpsc6430_t2/venv`
- pip: upgraded 24.0 → 26.2.1 (in venv)

## Step 1: PyTorch (CPU) — 2026-10-20 08:47 EDT
- Command: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Attempt 1: FAILED — connection timeout fetching from download-r2.pytorch.org ("didn't respond within 15 seconds"). Network/proxy hiccup, not a version problem.
- Attempt 2 (retry with `--timeout 120 --retries 5`): SUCCESS
- Installed: **torch 2.14.1+cpu** (plus sympy, networkx, fsspec, filelock, jinja2, MarkupSafe, typing-extensions)
- Verify: `import torch` OK; `torch.cuda.is_available()` = False (CPU-only, expected)

## Step 2: MuJoCo + robosuite — 2026-10-20 ~08:52 EDT
- Command: `pip install mujoco robosuite`
- Result: SUCCESS. Installed **mujoco 3.16.0**, **robosuite 1.5.2**, plus numpy 1.26.4, scipy 1.17.1, opencv-python 4.11.0.86, numba 0.68.0, Pillow 13.0.0, glfw, pyopengl, mink 0.0.5, etc.
- Verify: `import mujoco, robosuite, numpy` OK → mujoco 3.16.0, robosuite 1.5.2, numpy 1.26.4
- Non-fatal robosuite warnings on import (logged as setup notes):
  - "No private macro file found! ... run setup_macros.py" (optional)
  - "Could not import robosuite_models. Some robots may not be available." (optional package)
  - "Could not load the mink-based whole-body IK ... GR1 robot" (not needed for LIBERO/Panda)

## Step 3: LIBERO clone — 2026-10-20 ~08:56–09:08 EDT
- Command: `git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO`
- Result: SUCCESS (slow network, ~12 min). Repo at `~/workspace/cpsc6430_t2/LIBERO`.
- NOTE — `LIBERO/requirements.txt` pins 2022-era versions that are INCOMPATIBLE with this Python 3.12 environment: `numpy==1.22.4` (no cp312 wheels), `robosuite==1.4.0` (we run 1.5.2), `transformers==4.21.1`, `gym==0.25.2`, `opencv-python==4.6.0.66`, `robomimic==0.2.0`. We deliberately do NOT install requirements.txt verbatim; instead we install compatible current versions of only the packages the env/benchmark code needs.

## Step 4: LIBERO install — 2026-10-20 ~09:09 EDT
- Command: `cd LIBERO && pip install -e .`
- Result: pip reported SUCCESS ("Successfully installed libero-0.1.0") BUT the editable install is effectively broken: `setup.py` declares `install_requires=[]`, and the generated `__editable___libero_0_1_0_finder.py` has an EMPTY package MAPPING, so `import libero` failed from outside the repo dir (ModuleNotFoundError) and only "worked" from inside the repo as a namespace package (libero.__file__ = None).
- Fix applied: added a classic path file `venv/lib/python3.12/site-packages/libero_dev.pth` containing `/home/hatch/workspace/cpsc6430_t2/LIBERO`. After this, `libero.libero` resolves to the real package in the repo (data files stay in place).
- Next missing deps surfaced on import: `yaml` (pyyaml) — setup.py declares no dependencies, so runtime deps must be installed manually. Installing: pyyaml, bddl, easydict, hydra-core, einops, cloudpickle, future, matplotlib (current versions).

## Step 5: Dependency resolution saga — 2026-10-20 ~09:15–09:40 EDT
- `pip install pyyaml bddl easydict hydra-core einops cloudpickle future matplotlib`: SUCCESS (pyyaml 6.0.3, bddl 3.6.0, easydict 1.13, hydra-core 1.3.7, einops 0.8.2, cloudpickle 3.1.2, future 1.0.0, matplotlib 3.11.2) BUT it silently upgraded **numpy 1.26.4 → 2.5.4**, which conflicts with mink (`numpy<2`) and made the LIBERO import test appear to hang.
- Fix: `pip install numpy==1.26.4` (back to the robosuite-resolved version), then `pip install contourpy==1.2.1` (contourpy 1.4 needed numpy>=2). After this, `pip check` reports **"No broken requirements found"** and robosuite imports in seconds.
- First-import gotcha: `from libero.libero import benchmark` prompts interactively on first run — "Do you want to specify a custom path for the dataset folder? (Y/N):" — and writes `~/.libero/config.yaml`. Answered "N" (defaults). Config: benchmark_root/bddl_files/init_states/assets under `LIBERO/libero/libero/`, datasets at `LIBERO/libero/datasets` (not downloaded yet — expected warning "[Warning]: datasets path ... does not exist!").
- Import then failed deeper: `libero.libero.envs.bddl_base_domain` does `from robosuite.environments.manipulation.single_arm_env import SingleArmEnv` → **ModuleNotFoundError under robosuite 1.5.2** (module removed in 1.5). This LIBERO checkout needs the robosuite 1.4 API.
- Fix: `pip install robosuite==1.4.1` (matches LIBERO requirements.txt pin robosuite==1.4.0). SUCCESS.

## Step 6: Headless rendering (GL) — 2026-10-20 ~09:40–10:00 EDT
- With robosuite 1.4.1, import failed at `robosuite/utils/binding_utils.py` → `from mujoco.egl import egl_ext` → PyOpenGL `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` — this VM has **no libEGL and no libOSMesa** (only libGL/GLX), and robosuite's default macros (`MUJOCO_GPU_RENDERING=True`) force `MUJOCO_GL=egl`.
- Attempted system fix: `sudo apt-get install libegl1 libosmesa6` → initially "Unable to locate package"; `apt-get update` then hung >7 min with no output (Ubuntu mirrors effectively unreachable from this VM) — abandoned.
- Workaround found by reading binding_utils: macros do NOT override `MUJOCO_GL` if it is `osmesa` or `glx`; with `MUJOCO_GL=glx` robosuite falls back to the GLFW context module, which imports cleanly. We set `MUJOCO_GL=glx` in the environment of every run.
- LIBERO envs also need `gym`: `pip install gym==0.25.2` (the pinned version) SUCCESS.
- Env creation then progressed to `_reset_internal`, where robosuite 1.4.1 unconditionally creates an offscreen render context (`has_offscreen_renderer=True` is hardcoded in LIBERO's OffScreenRenderEnv) → initially `mujoco.FatalError: gladLoadGL error` (no display).
- Fix: **Xvfb is installed on this VM** (`/usr/bin/Xvfb`) — running under `xvfb-run -a` gives GLFW/GLX a virtual display and the render context initializes (Mesa software rendering).
- Next failure under Xvfb: `binding_utils.get_joint_qpos_addr` → `assert joint_type in (mjJNT_HINGE, mjJNT_SLIDE)` AssertionError — **mujoco 3.16 is too new for robosuite 1.4.1's binding layer**.
- Fix: `pip install mujoco==3.1.6` (robosuite-1.4.1-era pairing). SUCCESS — env constructs, resets, steps.

## Step 7: Smoke test — 2026-10-20 ~10:05 EDT — ✅ PASSED
- Script: `~/workspace/cpsc6430_t2/smoke_test.py`; full output saved in `smoke_test_output.txt`
- Command: `xvfb-run -a ./venv/bin/python smoke_test.py` (with MUJOCO_GL=glx set inside the script)
- Benchmark suites available: libero_10, libero_100, libero_90, libero_goal, libero_object, libero_spatial
- LIBERO-Goal: 10 tasks. Task 0: `open_the_middle_drawer_of_the_cabinet` ("open the middle drawer of the cabinet"). Task 1: `put_the_bowl_on_the_stove` ("put the bowl on the stove") — two tasks with different language goals, as the T2 plan requires.
- Env: OffScreenRenderEnv created from the task's BDDL file; `env.seed(42)` + `reset()` OK.
- Observation keys (25): `robot0_proprio-state` (39,), `object-state` (56,), `robot0_eef_pos` (3,), `robot0_eef_quat` (4,), `robot0_joint_pos` (7,), `robot0_joint_vel` (7,), `robot0_joint_pos_cos/sin` (7,), `robot0_gripper_qpos` (2,), `robot0_gripper_qvel` (2,), plus per-object `<obj>_pos` (3,), `<obj>_quat` (4,), `<obj>_to_robot0_eef_pos` (3,), `<obj>_to_robot0_eef_quat` (4,) for akita_black_bowl_1, cream_cheese_1, plate_1, wine_bottle_1.
- Action space: **7 components**, each in [-1, 1] (6-DoF end-effector delta + gripper) — matches the T2 doc's "seven action components" to verify.
- One random step: OK (reward 0.0, done=False). `env.check_success()` → False (expected after one random action).
- Bonus check: camera observations also work under Xvfb — `use_camera_obs=True` gives `agentview_image` (128,128,3) and `robot0_eye_in_hand_image` (128,128,3).

## Final working version set (pip list, 2026-10-20)
- python 3.12.3 | torch 2.14.1+cpu | mujoco 3.1.6 | robosuite 1.4.1 | libero 0.1.0 (editable, repo) | numpy 1.26.4 | gym 0.25.2 | bddl 3.6.0 | hydra-core 1.3.7 | easydict 1.13 | einops 0.8.2 | cloudpickle 3.1.2 | pyyaml 6.0.3 | opencv-python 4.11.0.86 | scipy 1.17.1 | matplotlib 3.11.2
- Run recipe: `cd ~/workspace/cpsc6430_t2 && xvfb-run -a ./venv/bin/python <script>` with `MUJOCO_GL=glx` in the environment.

## Known remaining gaps (next actions)
- LIBERO demonstration datasets NOT downloaded yet (datasets dir missing) — needed for Checkpoint 2 behavior cloning; download via LIBERO's download scripts when network allows.
- apt mirrors unreachable from this VM → could not install libegl1/libosmesa6; Xvfb+GLX is the working substitute. On a GPU machine, prefer MUJOCO_GL=egl.
- LIBERO's `requirements.txt` was intentionally NOT installed verbatim (2022 pins incompatible with Python 3.12); robomimic/transformers/wandb (training-time deps) not yet installed — add when starting Checkpoint 2 training.
