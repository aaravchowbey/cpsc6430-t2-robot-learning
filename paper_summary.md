# Paper Summary — Grid Sampler (GridS)

**Paper:** Feng, Zhao, Ma, Xia, Du, Wang, Xu — "See What Matters: Differentiable Grid Sample Pruning for Generalizable Vision-Language-Action Model," ICML 2026, PMLR 306:30552–30578.
**Source used:** Official PMLR PDF (proceedings.mlr.press/v306/feng26x.html → raw PDF, 27 pp., saved as `paper_gridsampler.pdf`), read in full — method (§3), experiments (§4), and Appendices A–K. Two implementation specifics (PyTorch `F.grid_sample`, coordinate MLP, default K) cross-checked against the official code repo, github.com/Fediory/Grid-Sampler.
**Purpose:** CPSC 6430 / T2 Checkpoint 1 literature section. The course project is **simulation-only**; sim vs. real parts of the paper are flagged throughout.

---

## 1. Problem & main contributions

**Problem.** Vision-Language-Action (VLA) policies are expensive because the vision encoder emits a dense, uniform token grid (e.g., 16×16 = 256 tokens per view) and the downstream Transformer's cost scales quadratically in sequence length, while most tokens are task-irrelevant background. Existing visual-token pruning doesn't fix this well:

- *Semantic pruning* (e.g., SparseVLM) keeps salient object bodies but discards geometrically critical, semantically subtle regions — edges, contact points.
- *Dynamic/training-free pruning* (FastV, VLA-Cache, EfficientVLA) is non-differentiable, applied post-hoc at test time, and degrades success rate.
- *All* discrete methods select from a fixed patch grid, so when a grasp point falls between patches they incur spatial quantization error and lose sub-patch precision.

**Contribution.** GridS reformulates compression as **geometry-aware, continuous token resampling inside the vision pathway**: instead of dropping patches, a small learned module predicts a minimal set of continuous 2D coordinates and extracts features there by differentiable bilinear interpolation. It is plug-and-play (sits between vision encoder and Transformer) and is trained jointly, end-to-end, with the policy.

**Headline claims.**
- Fewer than 10% of the original visual tokens retained (256 → 16 per view, i.e. 6.25%; also tested at K = 4, and K = 1 as a stress test).
- **76% FLOPs reduction with no success-rate degradation** on LIBERO — in fact small gains (+1.6 avg SR for π0 at K = 16) — described as the lowest feasible visual token count reported to date.
- ~1.2× single-batch inference speedup, up to ~3.2× at batch 128; training accelerated 3.4× (π0) / 2.9× (π0.5) per step.
- Large out-of-distribution gains: +28.6% OOD success rate in real-world tests (§4.2), attributed to an information-bottleneck / spatial-regularization effect — pruning forces the policy onto causal geometry instead of background shortcuts.

## 2. Method pipeline

GridS is inserted **between the (ViT) vision encoder and the downstream Transformer** of an existing VLA. The vision encoder is a standard frozen-style ViT (paper names SigLIP or DINOv2 as typical); an image becomes a dense feature grid T_dense ∈ R^{H×W×C} (e.g., 16×16 = 256 tokens). GridS then runs in two stages (Fig. 3 of the paper):

1. **Global Coordinate Prediction.** The dense feature map is collapsed by **global average pooling** into a single context vector z ∈ R^C (Eq. 1). A **lightweight MLP** maps z to **K sets of continuous, normalized coordinates** P = σ(MLP(z)) ∈ [0,1]^{K×2}, with a sigmoid keeping coordinates in-bounds. The predictor therefore sees the whole scene before deciding where to sample — sampling is task/scene-conditioned, not per-patch scoring. K ≪ H×W is a fixed, predefined budget.

2. **Grid Sampling with Geometry Injection.** For each predicted coordinate P(x, y):
   - **Differentiable bilinear sampling (§3.3):** the sampled feature is the weighted sum of the four nearest integer-grid neighbors, with weights ω₁=(1−Δx)(1−Δy), ω₂=Δx(1−Δy), ω₃=(1−Δx)Δy, ω₄=ΔxΔy from the fractional offsets (Eqs. 2–3). This gives sub-patch features — e.g., a point exactly on a patch boundary gets balanced contributions from both sides instead of a forced either/or choice. Because the weights are linear in (x, y), the operation is **fully differentiable w.r.t. the coordinates** (gradient in Eq. 4): the task loss literally moves sampling points toward locations that reduce action-prediction error. The coordinate MLP is trained by the downstream task, not by any heuristic.
   - **Coordinate injection:** bilinear sampling destroys the original 2D grid structure, so a **Coordinate Encoder** maps each predicted coordinate P into a position embedding E_pos ∈ R^{K×C}, which is **added** to the sampled features. The result is the sparse visual token set T_spa ∈ R^{K×C}, concatenated with the language and action tokens and fed to the downstream Transformer as usual. Ablating coordinate injection costs −3.6% SR; replacing bilinear with nearest-neighbor (non-differentiable) costs −4.1%.

**Backbones it is demonstrated on:** π0 and π0.5 (flow-matching VLAs), SmolVLA (autoregressive-style, used for the real-robot work), and X-VLA (RoboTwin appendix experiment). In the official code, the PyTorch version is an `ActiveTokenSampler` (LeRobot fork) implemented with `F.grid_sample` plus an (optional) coordinate MLP; the JAX π0/π0.5 version lives in openpi configs with `grid=True`; default K = 16.

## 3. Inputs, outputs, supervision, loss

- **Inputs:** RGB camera images (multi-view in the real-world setup: third-person + wrist; the LIBERO viewpoint-loss ablation confirms training with base + wrist cameras), a language instruction, and whatever proprio/state tokens the backbone normally consumes — GridS itself **only transforms the visual tokens** and does not touch language, state, or the action head. Image resolution: 224×224 for π0/π0.5, 512×512 for SmolVLA.
- **Outputs:** GridS outputs K sparse visual tokens (K×C). The VLA then outputs actions exactly as the base model does — action chunks of size 10 (π0/π0.5) or 50 (SmolVLA).
- **Supervision & loss:** GridS is trained with **only the primary task loss of the backbone** (the flow-matching loss for π0/π0.5; SmolVLA's own policy loss). There is **no auxiliary loss, no ground-truth attention/saliency supervision, no coordinate regularizer** mentioned anywhere. Training is **end-to-end joint fine-tuning** in a single stage — GridS is part of the model during standard VLA fine-tuning, in explicit contrast to training-free pruners bolted on after training. Optimizer AdamW; π0: LR 5e-5, batch 32, 30k steps; π0.5: LR 5e-5, batch 128, 30k steps; SmolVLA: LR 1e-4, batch 16, 50k steps (full table in §7 notes).
- **Consequence of "task-loss only":** there is no explicit pressure keeping sampled points spread out or interpretable; where points land is entirely emergent (visualizations show them concentrating on gripper/target affordances).

## 4. Datasets

- **LIBERO (simulation; Liu et al. 2023)** — the main benchmark. Four suites (Spatial, Object, Goal, Long), 10 tasks each, standard demonstration data (sim-collected teleoperation demonstrations distributed with the benchmark). GridS fine-tunes π0/π0.5 on the suites and evaluates in the MuJoCo-based LIBERO simulator.
- **LIBERO-PLUS (simulation; appendix)** — robustness extension: 10,030 evaluation tasks, 7 perturbation dimensions (camera viewpoint, robot initial state, language rewriting, lighting, background texture, sensor noise, object layout) × 21 low-level components, 5 difficulty levels, plus a 20,000+ trajectory generalized training set used for in-domain co-training.
- **ALOHA (simulation; §4.1.2)** — MuJoCo-based bimanual sim benchmark: Transfer Cube and Bimanual Insertion, **50 demonstrations per task** (scripted policies or human teleoperation), 400-step (8 s) episode horizon.
- **RoboTwin 2.0 (simulation; appendix, added during rebuttal)** — one task ("Place Bread Skillet") with X-VLA.
- **Real-robot dataset (REAL — outside course scope)** — self-collected on a LeRobot SO-100 arm via leader–follower teleoperation: Pick & Place 83 episodes, Transfer Pen 74 episodes, Stack Cubes 75 episodes (50 right-side / 25 left-side targets). RGB only (RealSense D435 third-person + D405 wrist), despite depth hardware being available.

## 5. Evaluation protocol

- **Tasks/splits:** There is **no train/test split in the ML sense** for LIBERO: models are fine-tuned per suite on that suite's demonstrations and evaluated on the benchmark's tasks with held-out initial states/episodes (the standard LIBERO protocol; the paper does not restate rollout counts for LIBERO — see §7). ALOHA protocol is stated precisely: 3 random seeds, ≥300 evaluation episodes over 100 unique initializations across the two tasks. Real-world: in-distribution tasks plus **21 OOD scenarios** in 7 perturbation categories (cluttered background, novel target instances, removed training priors, unknown distractors, spatial extrapolation, adversarial layouts, compound).
- **Baselines:** the dense backbones themselves (π0, π0.5, SmolVLA, plus OpenVLA, CogACT, OpenVLA-OFT numbers for context); training-free pruners **FastV, SparseVLM, VLA-Cache** applied to π0; and — most useful as controls — two **equal-budget downsampling baselines trained under the identical fine-tuning protocol at K = 16** (Appendix F): **Random Sampling** and **Top-K saliency pruning**.
- **Metrics:** task **success rate (%)** (primary); **FLOPs**, **visual token count**, **task completion/execution time and inference latency**, **training time per step**; environment return for ALOHA. Efficiency measured on a single RTX Pro6000 GPU with 32 language tokens, timing protocol following VLASH.
- **Headline results (LIBERO, Table 1):**
  - π0 dense: 256 tokens, 216.0 GFLOPs, 8.17 s, avg SR **94.4** (Goal suite 96.0).
  - π0 + GridS₁₆: 16 tokens, 51.7 GFLOPs (−76%), 6.05 s, avg SR **96.0 (+1.6)**; Goal **96.4**. GridS₄: avg 95.5 (+1.1), Goal 96.4 — a +1% LIBERO gain with only 4 tokens is highlighted in the intro.
  - π0.5 dense avg 96.7 → +GridS₁₆ **97.7 (+1.0)** (Goal 97.6 → 98.4), +GridS₄ 96.7 (±0.0).
  - Training-free pruners on π0 all lose: FastV −1.5, VLA-Cache −0.6, SparseVLM −4.6 avg SR.
  - Equal-K controls (App. F, K = 16): Random Sampling avg **87.8**, Top-K **90.5**, **GridS 96.0** — the gain is from the differentiable geometry-aware mechanism, not from token reduction per se.
  - Extreme compression (App. G): π0.5 + GridS with **K = 1** still averages **96.6** vs 96.7 dense.
  - Viewpoint-loss ablation (App. D): trained with both cameras, tested wrist-only — GridS (K=16, token masking) Spatial 0.92 vs baseline 0.66, Long 0.70 vs 0.30; wrist-only performance vs K is strongly non-monotonic (K=4 strong, K∈{6,8,10} dips, recovers ≥12).
  - LIBERO-PLUS (K = 32): in-domain co-trained GridS is within −1.1…−3.0 overall of the dense model; zero-shot OOD it roughly ties overall (Spatial −0.4, Goal −0.2) with big wins on camera viewpoint (+19.4 Spatial, +10.8 Long) and lighting/noise, and losses on robot-initial-state (−8.8…−14.0) and language-rewriting (−6.6…−12.0) perturbations.
  - ALOHA sim: π0 + GridS₁₆ avg SR **87.0** vs 86.3 dense; Insertion (Human demos) 64.2 vs 56.7 (**+7.5**).
  - RoboTwin: X-VLA + GridS **76%** vs 12% fine-tuned X-VLA baseline on Place Bread Skillet (single task, preliminary).
- **Compute:** fine-tuning on RTX Pro6000-class hardware (efficiency eval), batch 32 (π0) / 128 (π0.5), 30k steps; real-world training on a consumer RTX 3090 (24 GB).

## 6. Real-robot experiments — YES (flagged: outside course scope)

Section 4.2 + Appendices A/C and Table 3 / Fig. 5 are **real-robot**: a physical LeRobot SO-100 6-DoF arm, SmolVLA policy, RTX 3090 workstation, self-collected teleoperated data (§4 above).

- In-distribution: GridS beats dense SmolVLA on all three tasks; Stack Cubes **60.0% vs ~7.6% (+52.4)**; average success gain over baseline reported as +22.2%; execution time reduced by up to ~3.3 s (≈1.23× inference speedup on consumer GPU).
- OOD: across 21 scenarios GridS retains capability where the baseline collapses (baseline scores 0% OOD on Transfer Pen and Stack Cubes); **+28.6% OOD success rate** headline.
- Token-budget ablation on real Stack Cubes (SR / OOD SR): K=4 → 0.0/0.0; K=8 → 28.5/19.1; **K=16 → 61.9/38.1 (optimum)**; K=32 → 19.0/0.0; dense baseline (64 tokens) → 9.5/0.0. A clear **inverted-U**: too many tokens reintroduce background noise (at K=32 the policy stacks but fails to release the gripper); too few starve geometry.

**Sim parts of the paper (course-usable):** all of LIBERO (§4.1.1, Tables 1, 6–7, 10–12), ALOHA-sim (§4.1.2), LIBERO-PLUS (App. E.1), RoboTwin (App. E.2). **Real parts (exclude / cite only as motivation):** §4.2, the real-world halves of §4.3.1 and §4.4 visuals, App. A, App. C, App. K's real stacking sequence. Note the paper's single most dramatic number (+52.4 on Stack Cubes, +28.6% OOD) comes from the real-robot section — sim-only reporting must use the LIBERO/ALOHA numbers instead.

## 7. Unstated / ambiguous details (reproduction must guess)

- **Coordinate Encoder architecture:** the paper only says a "Coordinate Encoder" maps P → E_pos. No layer sizes, no statement of whether it's an MLP, sinusoidal encoding, or learned embedding. (Code: an optional coordinate MLP in `ActiveTokenSampler`; dimensions still config-level.)
- **Coordinate-predictor MLP size:** described only as "lightweight" / "Base MLP." The ablation scales it 10×/100× (+0.2% either way) without ever giving the base width/depth.
- **K per view vs. total:** Fig. 1 shows "2 × 256 → 2 × 16" (per-view), while Table 1 lists π0 at 256 visual tokens total and GridS at 16. Whether K is per camera view or across views is not pinned down in the text — matters for any multi-camera reproduction.
- **LIBERO evaluation rollouts:** episodes per task, seeds for the main LIBERO table, and the exact initial-state protocol are **not stated** (only the ALOHA protocol — 3 seeds, ≥300 episodes, 100 inits — is specified).
- **Fine-tuning data volume on LIBERO:** demonstrations actually used per task/suite are not stated (presumably the full standard LIBERO demo sets, but never confirmed in the text).
- **Frozen vs. fine-tuned vision encoder:** §3.1 says standard VLAs "typically employ a frozen ViT," while §3.4 says GridS is "jointly optimized with the VLA policy during standard fine-tuning." Whether encoder weights move is ambiguous.
- **Sampling-coordinate conventions:** conversion from sigmoid [0,1] outputs to feature-map indices (scaling, align-corners behavior) is unspecified in the paper (code uses `F.grid_sample` defaults).
- **Per-frame prediction:** coordinates are implicitly re-predicted per observation, but temporal behavior/consistency is never discussed.
- **Proprioception:** how state tokens interact with the pruned visual stream is not discussed at all (GridS is vision-only by design).
- **Hyperparameter table quirks (App. B, Table 5):** several cells are merged/ambiguous across models (e.g., warmup "10,000/1,000", decay LR "5e-5/2.5e-6"); betas [0.9, 0.95] and weight decay 1e-10 appear to apply to π0-family only. SmolVLA: 512×512 inputs, action chunk 50, grad-clip 10 vs 1.0 for π0-family.
- **Real-world eval counts:** trials per task/OOD scenario are not given (only 21 scenarios and dataset sizes), so the real-world SRs can't be given confidence intervals.
- **FLOPs accounting:** delegated to the VLASH protocol with a fixed 32 language tokens; not self-contained.
- **Known limitation relevant to reproduction:** GridS **underperforms the dense baseline by 8.3% under LoRA/PEFT** (App. J) — it presumes full fine-tuning capacity to re-align attention to the resampled token distribution. Also, K is static per model; no adaptive budget.
- **Component ablation reference points (App. H, Table 12, ΔSR vs default GridS):** conv-downsample pooling −4.4; global max pooling −0.3; SAM-guided sampling −6.2; no coordinate injection −3.6; nearest-neighbor sampling −4.1.

## 8. Course-adaptation notes (paper protocol → 2-task LIBERO-Goal course reproduction)

- **Scope collapse:** the paper evaluates 40 LIBERO tasks (4 suites) + LIBERO-PLUS's 10,030 perturbed tasks + ALOHA + RoboTwin + real robot. The course version uses **2 LIBERO-Goal tasks**; the only directly comparable published numbers are the LIBERO-Goal column (π0: 96.0 dense → 96.4 at K=16 and K=4; π0.5: 97.6 → 98.4 at K=16). Expect the course write-up to cite suite-level claims but only reproduce Goal-level evidence.
- **Backbone scale:** the paper fine-tunes large *pre-trained* VLAs (π0/π0.5, SmolVLA) whose visual/language grounding is already learned; GridS only has to learn *where to look*. A small course policy trained from ~60–100 demonstrations must learn perception, language grounding, and the coordinate predictor simultaneously, from the task loss alone — the paper offers no initialization, curriculum, or auxiliary-loss tricks for this regime because it never operates in it. Absolute success rates will not transfer; only the *relative* dense-vs-GridS comparison is meaningful.
- **Data volume & splits:** the paper uses the benchmark's full demonstration sets and its fixed initial-state evaluation (no explicit split). The course protocol — 60–100 sim demos total (≥30/task), ~70/15/15 whole-episode / initial-state-group splits, **20 held-out rollouts per variant per task** — is a course invention, stricter and more explicit than the paper's; it should be presented as such, not attributed to the paper.
- **The three course variants map cleanly:** vision-only behavior cloning ≈ a no-language ablation (not in the paper — the paper never removes language); dense image-language policy = the paper's dense baseline; learned GridS on the same backbone = the paper's treatment. **"Fixed resizing alone is not GridS"** in course terms = the sampler must be *learned coordinates + bilinear sampling + coordinate injection*, per §2.
- **The course's equal-budget uniform-sampling ablation has a direct paper analogue:** Appendix F's Random Sampling control at matched K = 16 and matched fine-tuning (87.8 vs GridS 96.0 avg SR). This is the right citation to justify the ablation design — and Top-K pruning (90.5) is available as an optional second control.
- **Choosing K:** copy the *ratio*, not the number. The paper's headline is K = 16 out of 256 tokens/view (6.25%); on a small policy with a much smaller feature grid, K should be set as a similar fraction and swept. Respect the inverted-U evidence: K too small starves geometry (real-world K=4 → 0%), K too large re-admits noise (real-world K=32 collapse; wrist-only LIBERO sweep is non-monotonic in K) — a small K-sweep is cheap insurance and mirrors the paper's own ablations.
- **Full fine-tuning only:** do not pair GridS with adapter/LoRA-style training in the course reproduction (paper: −8.3% vs dense under LoRA, App. J).
- **Metrics portability:** success rate and token count transfer directly. Latency/FLOPs do not: the paper's FLOPs use an RTX Pro6000 + VLASH protocol with 32 language tokens, and its own limitation section notes the wall-clock inference gain is only ~1.2× at batch 1 despite −76% FLOPs (fixed kernel overhead) — the course should report *relative* latency/token reduction on its own hardware and not promise the paper's 3.4× (that figure is training throughput at batch 128).
- **Provenance rule fit:** everything the course needs is in the paper's sim sections (LIBERO/ALOHA-sim/LIBERO-PLUS/RoboTwin). The real-robot SO-100 results, its teleoperated dataset, and its OOD scenario suite are excluded by the course's simulation-only rule — cite them only as external motivation, and do not quote +52.4% / +28.6% as if they were simulation results.
