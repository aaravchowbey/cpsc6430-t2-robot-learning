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

- PyTorch 2.14.1+cpu · MuJoCo 3.1.6 · robosuite 1.4.1 · LIBERO 0.1.0
- Observations: 25 keys (proprio state 39-d, object state 56-d, joint/gripper state, per-object features) plus 128×128 agentview and eye-in-hand images
- Action space: 7 components, each in [-1, 1] (6-DoF end-effector delta + gripper)
- Rendering on this headless machine uses software GL (Xvfb), as it has no GPU EGL

## Status — Checkpoint 1

Simulator stack installed and verified. A seeded smoke test on LIBERO-Goal task 0 passed: environment creation, `seed(42)` + reset, a random step, and `check_success()` behaving as expected. See `SETUP_STATUS.md` for the summary, `setup_log.md` for the full install log, and `smoke_test_output.txt` for the raw test output.

## Files

- `smoke_test.py` — LIBERO-Goal environment smoke test
- `smoke_test_output.txt` — captured smoke-test output
- `setup_log.md` — environment setup log
- `SETUP_STATUS.md` — setup status summary
- `paper_summary.md` — structured summary of the Grid Sampler paper
