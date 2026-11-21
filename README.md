# CPSC 6430 T2 — Simulation-Based Robot Learning

**Group:** Aarav Chowbey, Owen Rogers, Angelo Rodriguez
**Course:** CPSC 6430 Machine Learning · Topic T2: Simulation-Based Robot Learning · Fall 2026

## Project

Reproducing the **Grid Sampler** from Feng et al., *"See What Matters: Differentiable Grid Sample Pruning for Generalizable Vision-Language-Action Model"* (ICML 2026), and adding it to a small language-conditioned policy. All training and evaluation use simulated data only.

**Tasks (LIBERO-Goal):**
0. "open the middle drawer of the cabinet"
1. "put the bowl on the stove"

**Planned comparison** — three variants on one backbone:
1. Vision-only behavior cloning (no language input)
2. Dense image–language policy (all visual tokens)
3. Dense policy + learned Grid Sampler (scaled by the paper's ~6.25% token ratio)

**Data/evaluation plan:** 60–100 simulated demonstrations (≥30 per task), 70/15/15 whole-episode splits, predeclared success criteria, 20 held-out rollouts per variant per task (120 total). Primary metric: success rate. Ablation: uniform sampling at equal token budget. Improvement: per-view token-budget allocation across the agentview and eye-in-hand cameras.

## Environment

- Current Windows installation: Python 3.12.10 · PyTorch 2.15.0+cpu · MuJoCo 3.1.6 · robosuite 1.4.1 · LIBERO 0.1.0 · NumPy 1.26.4
- Observations: 25 keys (proprio state 39-d, object state 56-d, joint/gripper state, per-object features) plus 128×128 agentview and eye-in-hand images
- Action space: 7 components, each in [-1, 1] (6-DoF end-effector delta + gripper)
- Windows rendering uses GLFW. The original headless Linux setup used software GL through Xvfb and GLX.

## Status — Checkpoint 1

Simulator stack installed and verified. A seeded smoke test on LIBERO-Goal task 0 passed: environment creation, `seed(42)` + reset, a random step, and `check_success()` behaving as expected. See `SETUP_STATUS.md` for the summary, `setup_log.md` for the full install log, and `smoke_test_output.txt` for the raw test output.

The test also passed on the local Windows installation on November 17, 2026, ending with `SMOKE TEST DONE`. The random action returned reward `0.0`, `done=False`, and `check_success()=False`, as expected; this is a simulator check, not evidence of a trained policy completing the task. The existing setup logs and captured output describe the earlier Linux run.

## Setup for teammates

Use Git and Python 3.12. The checked-in policies run on CPU. Rendering requires a working OpenGL desktop or a virtual display on Linux. Downloading the full LIBERO-Goal suite takes approximately 6.4 GB.

From a fresh clone:

```text
git clone --recurse-submodules https://github.com/aaravchowbey/cpsc6430-t2-robot-learning.git
cd cpsc6430-t2-robot-learning
```

In an existing checkout, run `git submodule update --init` from the project root. This fetches the recorded LIBERO revision.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
$projectPython = '.venv/Scripts/python.exe'
& $projectPython -m pip install -r requirements.txt
& $projectPython setup_libero.py
$env:LIBERO_CONFIG_PATH = (Resolve-Path .libero).Path
& $projectPython -m pip check
& $projectPython smoke_test.py
```

### Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python setup_libero.py
export LIBERO_CONFIG_PATH="$PWD/.libero"
.venv/bin/python -m pip check
.venv/bin/python smoke_test.py
```

On headless Linux, use `xvfb-run -a .venv/bin/python smoke_test.py` and apply the same prefix to replay and policy commands. The setup script exposes the pinned LIBERO checkout to the virtual environment and configures robosuite's Windows rendering compatibility inside that environment. Set `LIBERO_CONFIG_PATH` again in new terminals before running the Python simulator or downloader commands. The smoke test should end with `SMOKE TEST DONE`.

## Checkpoint 2 progress — November 17, 2026

The LIBERO-Goal demonstration dataset was downloaded from the upstream project's Hugging Face source, `yifengzhu-hf/LIBERO-datasets`. The downloader reported `[X] Dataset libero_goal is complete`. Other suites were not downloaded and are not required for the selected tasks.

Download the data from the repository root after setup:

```powershell
& $projectPython LIBERO/benchmark_scripts/download_libero_datasets.py --datasets libero_goal --use-huggingface --download-dir data
$dataDir = 'data/libero_goal'
```

On Linux, replace `& $projectPython` with `.venv/bin/python`. If you already have the data, set `$dataDir` to your own folder containing the HDF5 files. The project uses `open_the_middle_drawer_of_the_cabinet_demo.hdf5` and `put_the_bowl_on_the_stove_demo.hdf5`. Each stores episodes under `data/demo_*`. The checkpoints and frozen split manifests are included in the repository. Keep the downloaded HDF5 files out of Git.

### Run the bowl execution demo

From the repository root in Windows PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./run_bowl_demo.ps1
```

The launcher uses the project virtual environment and `data/libero_goal` by default. Pass `-DataDir 'path/to/libero_goal'` for an existing dataset or `-Python 'path/to/python.exe'` for another configured interpreter. The demo uses the dense checkpoint, validation episode `demo_1`, and seed 42. It prints the simulator outcome and saves a CSV and labeled MP4 to a fresh timestamped folder under `output/policies/execution_demo/`.

On Linux, run the same demo with the Python entry point:

```bash
demo_dir=$(mktemp -d)
.venv/bin/python run_policy.py --checkpoint output/policies/dense.pt --data-dir data/libero_goal --task 1 --episode demo_1 --seed 42 --max-steps 300 --results "$demo_dir/bowl_results.csv" --video-dir "$demo_dir/videos"
```

### Record three runs per model and task

With `evaluate_heldout.py` from the latest team checkout, run this from Windows PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./run_comparison_demo.ps1
```

The launcher records nine clips for each task. All three models use drawer starts 37–39 and bowl starts 30–32. Each rollout starts in its own simulator process. Find the MP4s and rollout CSVs under the timestamped folder printed at the start. If a run is interrupted, repeat the command with `-OutputDir 'path/to/the/timestamped/folder'` to skip runs that already have a result and video. Pass `-DataDir` and `-Python` to use an existing dataset or Python installation, as with the bowl launcher.

### Data and simulator progress

- Completed: inspect both selected HDF5 files and select all 100 demonstrations, 50 per task.
- Completed: create and verify frozen 70/15/15 episode split manifests; document grouping and the missing-session-ID limitation below.
- Completed: simulator smoke test and drawer state/action replay, including state restoration and simulator success checks.
- User reviewed the bowl replay video and reported that it looks correct. Its terminal success result has not been supplied in the chat; visual appearance alone does not establish the simulator success result.

Remaining data/simulator evidence for the final submission:

1. Retain the labeled replay MP4s for both tasks and capture each command's state-restoration and simulator-success output. Confirm the bowl replay's printed result.
2. Include example demonstration videos, split counts/manifests, simulator details, and the missing-session-ID limitation in the report and submission ZIP. The README does not replace the report or videos.

The shared data loader, three policy variants, and autonomous validation runner are available below. The final 20-rollout evaluation per variant per task still needs a common held-out start-state plan.

### Train and run the policies

Run from the repository root with the virtual environment Python and dataset path above. On Linux, use `.venv/bin/python` and pass your data path directly:

```powershell
$env:NUMBA_CACHE_DIR = Join-Path $env:TEMP 'libero-numba'
& $projectPython train_policy.py --data-dir $dataDir --smoke --epochs 35 --output output/policies/local
& $projectPython train_policy.py --data-dir $dataDir --variant all --epochs 20 --output output/policies/local

foreach ($variant in @('vision', 'dense', 'grid')) {
    & $projectPython run_policy.py --checkpoint "output/policies/local/$variant.pt" --data-dir $dataDir --results "output/policies/local/${variant}_validation.csv" --video-dir output/policies/local/validation --per-task 1 --max-steps 300
}
```

The loader follows the frozen manifests and pairs both post-action RGB frames at index `i` with action `i+1` in the same episode. It resizes each camera to 48×48 with area interpolation and scales image values to [0, 1] in the model. The full run has 8,411 training pairs and 1,844 validation pairs. The smoke run reuses 32 training pairs to check optimization.

Each policy uses a shared-weight CNN across the two camera views, learned camera and coordinate embeddings, attention pooling, and a seven-component action head. The vision variant uses RGB only. The other variants embed bag-of-words counts for the two fixed training instructions. The dense variants supply 32 visual tokens to attention. Grid Sampler predicts one normalized coordinate per camera from globally pooled features and the instruction, samples each feature map with differentiable bilinear interpolation, and adds a learned embedding of the sampled coordinates. All models start from random weights and train with AdamW, a batch size of 128, a learning rate of 0.001, and mean squared action error. The best validation loss determines the saved checkpoint.

The runner restores the specified initial scene and uses live camera images to predict every action. It stops at simulator success, environment termination, or 300 control steps. The CSV records outcomes, step counts, visual tokens, and CPU inference latency including image preprocessing. The optional MP4s show the policy name, task ID, episode, seed, and simulator success. Pass `--task 0 --episode demo_12` to rerun one specific start. Use a new results filename when repeating a run. The loop above runs two validation trials per variant. The demonstration test split has fewer than 20 independent starts per task, so these commands alone do not fulfill the final rollout requirement.

Initial CPU validation used drawer `demo_12` and bowl `demo_1`, both with seed 42. The checkpoints, loss logs, rollout CSVs, and labeled videos are in `output/policies/`.

| Policy | Selected epoch | Validation action MSE | Visual tokens | Drawer | Bowl |
|---|---:|---:|---:|---|---|
| Vision-only | 20 | 0.03277 | 32 | Timed out | Timed out |
| Dense image-language | 20 | 0.02546 | 32 | Timed out | Success at step 89 |
| Learned Grid Sampler | 18 | 0.03249 | 2 | Timed out | Timed out |

A separate dense drawer trial on validation episode `demo_15` (seed 43) also timed out. These trials establish that all variants execute autonomously and that the dense policy can complete the bowl task on this validation start. Autonomous drawer completion and the final held-out comparison are still pending.

Inspection found 50 demonstrations in each selected file (100 total). Both contain `actions`, `dones`, `obs`, `rewards`, `robot_states`, and `states`. Camera observations are named `agentview_rgb` and `eye_in_hand_rgb`; other observations include end-effector, gripper, and joint states. The inspected drawer episode has 138 actions and the bowl episode has 94, each with seven action components; lengths vary by episode.

### Replay an episode in the simulator

`replay_episode.py` restores the episode's recorded scene model, fixes local asset paths, checks starting-state restoration, and saves a two-camera MP4 with task, episode, seed, replay mode, and simulator-success labels.

```powershell
& $projectPython replay_episode.py --data-dir $dataDir --task 0 --episode demo_0
& $projectPython replay_episode.py --data-dir $dataDir --task 1 --episode demo_0
```

Videos are saved under `output/replays/`. The default `--mode states` restores successive recorded states for faithful visualization. To test stepping the simulator with recorded actions from the restored starting state, add `--mode actions`. Action replay can diverge when simulator versions or controller behavior differ from data collection. The printed seed identifies the reset before restoration; the recorded state determines the replay's actual starting scene.

The drawer task's `demo_0` state and action replays passed restoration and finished with simulator success `True` on Windows. These are demonstration replays, not autonomous learned-policy runs.

### Frozen episode splits

`splits/create_splits.py` created `splits/train.csv`, `splits/validation.csv`, `splits/test.csv`, and `splits/summary.json`, using seed 42 and all 100 demonstrations:

| Task | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Open middle drawer | 35 | 8 | 7 | 50 |
| Put bowl on stove | 35 | 7 | 8 | 50 |
| Total | 70 | 15 | 15 | 100 |

Each manifest identifies the source filename, episode, number of steps, initial-state hash, scene-group ID, and split. Files stay in the external data directory; episodes are not copied. The summary records dataset SHA-256 hashes and the grouping method. Commit these small manifests with the code and use the same assignments for all policy variants.

Grouping compares non-robot joint positions rounded to four decimals and fixed scene-body poses from the saved XML, excluding robot state and simulator time. Quaternion signs are normalized. All 50 episodes in each task have distinct starting-scene group IDs under this rule. The files contain no session identifiers, so session independence cannot be verified; near-matching scenes on rounding boundaries may also escape grouping. This limitation must be reported. Both inspected `demo_0` episodes were reserved for training.

To reproduce into a new folder (the script refuses to overwrite existing manifests):

```powershell
& $projectPython splits/create_splits.py --data-dir $dataDir --output-dir splits-reproduced --seed 42
```

Fit learned preprocessing on training episodes only, select models using validation episodes, and reserve test episodes for final testing. The 15 test demonstrations are separate from the requirement for 20 autonomous simulator rollouts per policy per task; rollout initial states and seeds still need a declared held-out evaluation protocol.

## Files

- `smoke_test.py` — LIBERO-Goal environment smoke test
- `replay_episode.py` — simulator replay and labeled video export for recorded demonstrations
- `splits/create_splits.py` — deterministic whole-episode split generation and source-data fingerprints
- `splits/` — frozen train, validation, and test manifests plus split metadata
- `smoke_test_output.txt` — captured smoke-test output
- `setup_log.md` — environment setup log
- `SETUP_STATUS.md` — setup status summary
- `paper_summary.md` — structured summary of the Grid Sampler paper

## AI Assistance

I, Owen Rogers, used OpenAI’s ChatGPT to help write the PowerShell demo launchers, the LIBERO setup script, add a single-rollout option to the evaluation script, and update the README with setup and demo instructions. 
## Submitted version

Release tag: checkpoint2 (https://github.com/aaravchowbey/cpsc6430-t2-robot-learning/tree/checkpoint2)
