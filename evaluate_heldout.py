"""Final held-out evaluation for the T2 LIBERO-Goal policies (simulation only).

Runs every trained policy variant autonomously from the SAME held-out start states:
LIBERO-Goal's official benchmark initial states 30..49 by default (--first-init 30, --n 20)
for each task. These benchmark states ship with LIBERO and are separate from the 100
demonstration episodes; this script checks that none of them matches a demonstration
start under the team's frozen scene-grouping rule before any rollout is run.

Place this file in the repository root (next to train_policy.py) and run from there.

Outputs (in --out-dir):
  heldout_starts.csv   held-out start-state manifest + leakage checks
  rollouts.csv         one row per rollout (resumable: completed rows are skipped)
  summary.csv / summary.md   success counts, rates, Wilson 95% CIs, latency, compute budget,
                       and paired exact McNemar comparisons between variants
  videos/              labeled MP4 per rollout ("AUTONOMOUS POLICY", task, init, seed, success)
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter

# Configure the GL backend before LIBERO/robosuite are imported (same rule as replay_episode.py).
if os.name == "nt":
    os.environ["MUJOCO_GL"] = "glfw"
else:
    os.environ.setdefault("MUJOCO_GL", "glx")
    if os.environ["MUJOCO_GL"] in ("egl", "osmesa"):
        os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "splits"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from create_splits import scene_group  # noqa: E402
from train_policy import Policy, read_manifest, resize_view  # noqa: E402

TASKS = {0: "open_the_middle_drawer_of_the_cabinet", 1: "put_the_bowl_on_the_stove"}
LIVE_CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
FIELDS = ("variant", "checkpoint", "checkpoint_epoch", "task_id", "task", "init_index",
          "init_state_sha256", "seed", "outcome", "success", "steps", "max_steps",
          "mean_inference_ms", "visual_tokens", "video")


# ---------------------------------------------------------------- helpers
def sha(state):
    return hashlib.sha256(np.asarray(state, dtype=np.float64).tobytes()).hexdigest()


def make_env(task):
    return OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        camera_heights=128, camera_widths=128, use_camera_obs=True,
    )


def load_init_states(task):
    # LIBERO stores these as pickled numpy arrays; newer torch needs weights_only=False.
    # (benchmark.get_task_init_states() fails on torch >= 2.6 for this reason.)
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return np.asarray(torch.load(path, weights_only=False), dtype=np.float64)


def object_vector(xml, state):
    """Non-robot joint positions in XML order (quaternion sign normalized), for near-match checks."""
    import xml.etree.ElementTree as ET
    if isinstance(xml, bytes):
        xml = xml.decode()
    root = ET.fromstring(xml)
    values, names, offset = [], [], 1
    for joint in root.find("worldbody").iter():
        if joint.tag not in ("joint", "freejoint"):
            continue
        name = joint.get("name", "")
        kind = "free" if joint.tag == "freejoint" else joint.get("type", "hinge")
        width = {"free": 7, "ball": 4, "hinge": 1, "slide": 1}[kind]
        if not name.startswith(("robot", "gripper")):
            part = np.array(state[offset:offset + width], dtype=np.float64)
            if kind in ("free", "ball"):
                q0 = 3 if kind == "free" else 0
                nz = np.flatnonzero(np.abs(part[q0:]) > 1e-10)
                if len(nz) and part[q0 + nz[0]] < 0:
                    part[q0:] *= -1
            values.append(part)
            names.append(name)
        offset += width
    return names, np.concatenate(values)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Policy(checkpoint["variant"], checkpoint["state_dict"]["instructions"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if checkpoint.get("smoke"):
        raise ValueError(f"{path} is a smoke-test checkpoint; use a full training checkpoint")
    return model, checkpoint


def compute_budget(model):
    params = sum(p.numel() for p in model.parameters())
    images = torch.zeros(1, 2, 3, 48, 48, dtype=torch.uint8)
    flops = float("nan")
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as counter, torch.inference_mode():
            model(images, torch.tensor([0]))
        flops = counter.get_total_flops()
    except Exception:  # FLOP counting is informational only
        pass
    return params, flops


# ---------------------------------------------------------------- held-out start manifest
def build_start_manifest(suite, args):
    rows = []
    for split in ("train", "validation", "test"):
        rows += read_manifest(args.splits / f"{split}.csv")
    demo_vectors = {}
    if args.data_dir:
        import h5py
        for row in rows:
            with h5py.File(args.data_dir / row["dataset"], "r") as source:
                episode = source[f"data/{row['episode']}"]
                state = episode["states"][0]
                if sha(state) != row["initial_state_sha256"]:
                    raise ValueError(f"Data file does not match frozen manifest: {row['episode']}")
                demo_vectors.setdefault(int(row["task_id"]), []).append(
                    (row["episode"], row["split"], object_vector(episode.attrs["model_file"], state)))
    starts = []
    for task_id in args.tasks:
        task = suite.get_task(task_id)
        if task.name != TASKS[task_id]:
            raise ValueError(f"Suite task {task_id} is {task.name}, expected {TASKS[task_id]}")
        init_states = load_init_states(task)
        if args.first_init + args.n > len(init_states):
            raise ValueError(f"Only {len(init_states)} benchmark init states for {task.name}")
        env = make_env(task)
        env.seed(args.seed)
        env.reset()
        xml = env.env.sim.model.get_xml()
        env.close()
        demo_hashes = {r["initial_state_sha256"] for r in rows if int(r["task_id"]) == task_id}
        demo_groups = {r["group_id"] for r in rows if int(r["task_id"]) == task_id}
        for index in range(args.first_init, args.first_init + args.n):
            state = init_states[index]
            entry = dict(task_id=task_id, task=task.name, init_index=index,
                         init_state_sha256=sha(state), scene_group=scene_group(xml, state),
                         seed=args.seed + index)
            entry["exact_match_any_demo"] = entry["init_state_sha256"] in demo_hashes
            entry["scene_group_match_any_demo"] = entry["scene_group"] in demo_groups
            if demo_vectors:
                names, vector = object_vector(xml, state)
                best = min(((float(np.linalg.norm(vector - v)), ep, sp)
                            for ep, sp, (n2, v) in demo_vectors[task_id] if n2 == names),
                           default=(float("nan"), "", ""))
                entry.update(nearest_demo=best[1], nearest_demo_split=best[2],
                             nearest_demo_object_l2=round(best[0], 6))
            starts.append(entry)
            if entry["exact_match_any_demo"] or entry["scene_group_match_any_demo"]:
                raise RuntimeError(f"Leakage: benchmark init {index} of {task.name} matches a demonstration start")
    return starts


# ---------------------------------------------------------------- rollout
def rollout(task, start, init_state, model, args, video_path):
    env = make_env(task)
    writer = None
    try:
        env.seed(start["seed"])
        env.reset()
        obs = env.set_init_state(init_state)
        # LIBERO's own evaluation lets objects settle with a few no-op steps (gripper open);
        # these are not policy actions and do not count toward the step budget.
        for _ in range(args.settle_steps):
            obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64))
        lines = (f"AUTONOMOUS POLICY: {model.variant}  |  task {start['task_id']}: {task.language}",
                 f"held-out benchmark init {start['init_index']}  |  seed {start['seed']}")
        if video_path:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (512, 352))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open video writer: {video_path}")

        def record(observation, step, success, repeat=1):
            if writer is None:
                return
            views = np.concatenate([observation[key][::-1] for key in LIVE_CAMERAS], axis=1)
            canvas = np.zeros((352, 512, 3), dtype=np.uint8)
            canvas[96:] = cv2.cvtColor(cv2.resize(views, (512, 256)), cv2.COLOR_RGB2BGR)
            colour = (80, 220, 80) if success else (255, 255, 255)
            texts = (*lines, f"step {step}/{args.max_steps}  |  simulator success = {success}")
            for y, text, c in zip((24, 52, 80), texts, ((255, 255, 255), (255, 255, 255), colour)):
                cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
            for _ in range(repeat):
                writer.write(canvas)

        success = bool(env.check_success())
        if success:
            return "already_success_at_start", 0, float("nan")
        record(obs, 0, success)
        elapsed, steps, done = 0.0, 0, False
        task_id = torch.tensor([start["task_id"]], dtype=torch.long)
        for steps in range(1, args.max_steps + 1):
            t0 = perf_counter()
            images = np.stack([resize_view(obs[key]) for key in LIVE_CAMERAS])
            images = torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(0)
            with torch.inference_mode():
                action = model(images, task_id)[0].numpy()
            elapsed += perf_counter() - t0
            obs, _, done, _ = env.step(action)
            success = bool(env.check_success())
            record(obs, steps, success)
            if success or done:
                break
        record(obs, steps, success, repeat=30)
        outcome = "success" if success else "terminated" if done else "timeout"
        return outcome, steps, 1000 * elapsed / max(steps, 1)
    finally:
        if writer is not None:
            writer.release()
        env.close()


# ---------------------------------------------------------------- summary
def summarize(out_dir, budgets):
    rows = list(csv.DictReader((out_dir / "rollouts.csv").open(encoding="utf-8")))
    variants = sorted({r["variant"] for r in rows}, key=lambda v: ("vision", "dense", "grid").index(v)
                      if v in ("vision", "dense", "grid") else 9)
    tasks = sorted({int(r["task_id"]) for r in rows})
    table = []
    for variant in variants:
        for task_id in [*tasks, "all"]:
            sel = [r for r in rows if r["variant"] == variant and (task_id == "all" or int(r["task_id"]) == task_id)]
            if not sel:
                continue
            k = sum(r["success"] == "True" for r in sel)
            lo, hi = wilson(k, len(sel))
            ok_steps = [int(r["steps"]) for r in sel if r["success"] == "True"]
            lat = [float(r["mean_inference_ms"]) for r in sel if r["mean_inference_ms"] not in ("", "nan")]
            b = budgets.get(variant, {})
            table.append(dict(
                variant=variant, task=("both" if task_id == "all" else TASKS[task_id]), n=len(sel),
                successes=k, success_rate=round(k / len(sel), 3), wilson95_low=round(lo, 3), wilson95_high=round(hi, 3),
                mean_steps_successful=(round(float(np.mean(ok_steps)), 1) if ok_steps else ""),
                timeouts=sum(r["outcome"] == "timeout" for r in sel),
                median_inference_ms=(round(float(np.median(lat)), 2) if lat else ""),
                visual_tokens=sel[0]["visual_tokens"], parameters=b.get("parameters", ""),
                forward_mflops=(round(b["flops"] / 1e6, 2) if b.get("flops") == b.get("flops") and b.get("flops") is not None else ""),
            ))
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    # Paired comparisons on identical start states.
    outcome = {(r["variant"], int(r["task_id"]), int(r["init_index"])): r["success"] == "True" for r in rows}
    pairs = []
    for i, a in enumerate(variants):
        for bvar in variants[i + 1:]:
            keys = [(t, n) for (v, t, n) in outcome if v == a and (bvar, t, n) in outcome]
            b_only = sum(outcome[(a, t, n)] and not outcome[(bvar, t, n)] for t, n in keys)
            c_only = sum(outcome[(bvar, t, n)] and not outcome[(a, t, n)] for t, n in keys)
            pairs.append((a, bvar, len(keys), b_only, c_only, mcnemar_exact(b_only, c_only)))
    md = ["| Variant | Task | Successes / n | Rate | Wilson 95% CI | Mean steps (succ.) | Median latency ms | Visual tokens | Params | MFLOPs/fwd |",
          "|---|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for t in table:
        md.append(f"| {t['variant']} | {t['task']} | {t['successes']}/{t['n']} | {t['success_rate']:.0%} | "
                  f"{t['wilson95_low']:.2f}–{t['wilson95_high']:.2f} | {t['mean_steps_successful']} | "
                  f"{t['median_inference_ms']} | {t['visual_tokens']} | {t['parameters']} | {t['forward_mflops']} |")
    md += ["", "Paired exact McNemar tests (same held-out starts, both tasks pooled):", "",
           "| A | B | paired starts | A-only successes | B-only successes | p |", "|---|---|---:|---:|---:|---:|"]
    md += [f"| {a} | {b} | {n} | {x} | {y} | {p:.3f} |" for a, b, n, x, y, p in pairs]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=Path, nargs="+",
                        default=[Path("output/policies") / f"{v}.pt" for v in ("vision", "dense", "grid")])
    parser.add_argument("--tasks", type=int, nargs="+", choices=(0, 1), default=[0, 1])
    parser.add_argument("--n", type=int, default=20, help="Held-out rollouts per variant per task")
    parser.add_argument("--first-init", type=int, default=30,
                        help="First benchmark init-state index. Official test block = 30..49 (declared before testing); "
                             "0..29 may be used for debugging/model selection")
    parser.add_argument("--max-steps", type=int, default=300, help="Timeout (policy steps), declared before testing")
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000, help="Rollout seed = seed + init index")
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--data-dir", type=Path, help="Optional: HDF5 folder, enables nearest-demo distance check")
    parser.add_argument("--out-dir", type=Path, default=Path("output/heldout_eval"))
    parser.add_argument("--videos", choices=("all", "none"), default="all")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--only-rollout", nargs=2, metavar=("VARIANT", "INIT"),
                        help="Run just this variant and start while keeping the full selected manifest")
    parser.add_argument("--summary-only", action="store_true", help="Rebuild summary from rollouts.csv")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    models = [(path, *load_policy(path)) for path in args.checkpoints]
    selected = None
    if args.only_rollout:
        variant, raw_index = args.only_rollout
        try:
            index = int(raw_index)
        except ValueError:
            parser.error("--only-rollout INIT must be an integer")
        if variant not in {model.variant for _, model, _ in models}:
            parser.error("--only-rollout VARIANT must match a selected checkpoint")
        if not args.first_init <= index < args.first_init + args.n:
            parser.error("--only-rollout INIT must be in the selected start range")
        selected = (variant, index)
    budgets = {}
    for path, model, _ in models:
        params, flops = compute_budget(model)
        budgets[model.variant] = dict(parameters=params, flops=flops)
    (args.out_dir / "compute_budget.json").write_text(json.dumps(budgets, indent=2) + "\n", encoding="utf-8")
    if args.summary_only:
        summarize(args.out_dir, budgets)
        return

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    starts = build_start_manifest(suite, args)
    manifest = args.out_dir / "heldout_starts.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(starts[0]))
        w.writeheader()
        w.writerows(starts)
    print(f"Held-out starts: {len(starts)} (no exact or scene-group match with any of the 100 demos) -> {manifest}",
          flush=True)

    results = args.out_dir / "rollouts.csv"
    done = set()
    if results.exists():
        done = {(r["variant"], int(r["task_id"]), int(r["init_index"]))
                for r in csv.DictReader(results.open(encoding="utf-8"))}
    new_file = not results.exists()
    init_cache = {}
    with results.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for start in starts:
            task = suite.get_task(start["task_id"])
            if start["task_id"] not in init_cache:
                init_cache[start["task_id"]] = load_init_states(task)
            init_state = init_cache[start["task_id"]][start["init_index"]]
            for path, model, checkpoint in models:
                key = (model.variant, start["task_id"], start["init_index"])
                if key in done or (selected is not None and (model.variant, start["init_index"]) != selected):
                    continue
                video = None
                if args.videos == "all":
                    video = args.out_dir / "videos" / f"{model.variant}_task{start['task_id']}_init{start['init_index']:02d}_seed{start['seed']}.mp4"
                t0 = perf_counter()
                outcome, steps, latency = rollout(task, start, init_state, model, args, video)
                writer.writerow(dict(
                    variant=model.variant, checkpoint=path.as_posix(), checkpoint_epoch=checkpoint["epoch"],
                    task_id=start["task_id"], task=task.name, init_index=start["init_index"],
                    init_state_sha256=start["init_state_sha256"], seed=start["seed"], outcome=outcome,
                    success=outcome == "success", steps=steps, max_steps=args.max_steps,
                    mean_inference_ms=round(latency, 3), visual_tokens=model.visual_tokens,
                    video=video.as_posix() if video else ""))
                f.flush()
                print(f"{model.variant:6s} task={start['task_id']} init={start['init_index']:02d} seed={start['seed']} "
                      f"outcome={outcome} steps={steps} latency_ms={latency:.2f} wall_s={perf_counter() - t0:.1f}",
                      flush=True)
    summarize(args.out_dir, budgets)


if __name__ == "__main__":
    main()
