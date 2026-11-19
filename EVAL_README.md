# Held-out evaluation (`evaluate_heldout.py`)

Final T2 evaluation: 20 autonomous rollouts per policy variant per task, all variants on the same held-out start states.

## Held-out protocol (declare in the report before showing results)

- **Start states:** LIBERO-Goal's official benchmark initial states, indices **30–49** for each task (20 per task). These ship with LIBERO (`libero/libero/init_files/libero_goal/*.pruned_init`). They are separate from the 100 demonstration episodes.
- **Leakage check (automatic):** before running any rollout, the script compares every held-out start against all 100 demo starts (train, validation and test). It checks for an exact state-hash match and for a match under the team's frozen scene-grouping rule (`splits/create_splits.py`). If anything matches, the script stops. All 50 benchmark starts per task passed both checks. With `--data-dir`, it also records each start's object-position distance to the nearest demo.
- **Rollout rule:** reset with `seed = 1000 + init index`, restore the benchmark state, then 5 no-op settle steps with the gripper open (LIBERO's own evaluation does the same; these don't count toward the budget). After that, the policy acts from live camera images. A rollout ends at simulator `check_success()` = True (**success**), environment termination, or **300 policy steps** (**timeout**).
- **Determinism:** the simulator and the policies are deterministic, so the independent unit is the **start state**, not the seed. Rerunning one start with another seed reproduces the same rollout. That's why 20 distinct starts are used.
- **Same starts for every variant:** this allows paired comparisons (exact McNemar test).
- Benchmark starts 0–29 may be used for debugging or model selection. **Don't** use 30–49 for any decision before the final run. *(Starts 0–3 of the bowl task were used for a dense-policy preview, and 0–1 of both tasks for a 25-step script check.)*

## Run (Windows, from the repository root)

Copy `evaluate_heldout.py` into the repository root next to `train_policy.py`, then:

```powershell
$projectPython = 'C:\Users\owenr\AppData\Local\Programs\Python\Python312\python.exe'
$dataDir = 'C:\Users\owenr\Desktop\libero-data\libero_goal'
$env:NUMBA_CACHE_DIR = Join-Path $env:TEMP 'libero-numba'

# Final evaluation: 3 variants x 2 tasks x 20 starts = 120 rollouts, labeled MP4 for each
& $projectPython evaluate_heldout.py --data-dir $dataDir --out-dir output/heldout_eval
```

- **Resumable:** if the run is interrupted, rerun the same command. Rows already in `rollouts.csv` are skipped.
- **Rebuild tables only:** `& $projectPython evaluate_heldout.py --summary-only --out-dir output/heldout_eval`
- **Checkpoints:** the default is `output/policies/{vision,dense,grid}.pt`. If the team retrains, pass the frozen checkpoints with `--checkpoints a.pt b.pt c.pt` and use a **new** `--out-dir`.
- **Linux:** set `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU software rendering).

## Outputs (`output/heldout_eval/`)

| File | Contents |
|---|---|
| `heldout_starts.csv` | Held-out start manifest: task, init index, state hash, scene group, seed, leakage flags, nearest demo |
| `rollouts.csv` | One row per rollout: variant, checkpoint epoch, task, init, seed, outcome, steps, mean inference ms, visual tokens, video path |
| `summary.md` / `summary.csv` | Successes/n, rate, Wilson 95% CI, mean steps on success, median latency, visual tokens, parameter count, MFLOPs per forward pass, paired McNemar tests |
| `compute_budget.json` | Parameters and FLOPs per variant |
| `videos/*.mp4` | Labeled "AUTONOMOUS POLICY" clips showing the task, held-out init, seed, step and simulator success |

## Runtime

Measured on a 2-core cloud CPU with software rendering: about 50 s per 300-step rollout, about 20 s per early success. The full 120-rollout run takes up to about 1.5 h on that machine, and should be faster on a laptop with GPU rendering.

## Notes

- LIBERO's `benchmark.get_task_init_states()` fails on torch ≥ 2.6 (`weights_only` error). The script loads the init files directly with `torch.load(..., weights_only=False)`. They're LIBERO's own upstream files.
- Latency is CPU inference including image preprocessing, averaged per rollout, and excludes simulator time. FLOPs come from `torch.utils.flop_counter` at 48×48 input and may not count `grid_sample`.

## Held-out evaluation (Checkpoint 2 final results)

`evaluate_heldout.py` runs each policy autonomously from LIBERO-Goal's official benchmark start states 30–49 (20 per task, same starts for every policy). Before any rollout, it checks that no start matches any of the 100 demo starts. Success = simulator `check_success()`; timeout = 300 policy steps.

```bash
# macOS (Linux: MUJOCO_GL=egl; Windows: no variable needed)
MUJOCO_GL=glfw python evaluate_heldout.py --out-dir output/heldout_eval
```

Outputs in `output/heldout_eval/`: `rollouts.csv` (one row per rollout), `heldout_starts.csv` (held-out start manifest + leakage checks), `summary.md` / `summary.csv` (success rates, Wilson 95% CIs, latency, tokens, parameters, paired McNemar tests), `compute_budget.json`, and labeled videos. The final run used one process per policy (`--checkpoints output/policies/<variant>.pt --out-dir output/heldout_eval/<variant>`), merged into one `rollouts.csv`, then `--summary-only`. If a run stops, rerun the same command; finished rollouts are skipped. About 10 s per rollout on a MacBook Pro (Apple Silicon, CPU).