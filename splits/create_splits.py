"""Create reproducible whole-episode LIBERO splits without copying datasets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import h5py
import numpy as np

TASKS = ["open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_stove"]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def scene_group(xml, state):
    """Read MuJoCo joint order from XML without loading rendering assets."""
    if isinstance(xml, bytes):
        xml = xml.decode()
    root = ET.fromstring(xml)
    scene = []
    offset, nv = 1, 0
    for joint in root.find("worldbody").iter():
        if joint.tag not in ("joint", "freejoint"):
            continue
        name = joint.get("name", "")
        kind = "free" if joint.tag == "freejoint" else joint.get("type", "hinge")
        width, velocities = {"free": (7, 6), "ball": (4, 3), "hinge": (1, 1), "slide": (1, 1)}[kind]
        if not name.startswith(("robot", "gripper")):
            values = state[offset:offset + width].copy()
            # q and -q describe the same orientation.
            if kind in ("free", "ball"):
                quat_start = 3 if kind == "free" else 0
                quat = values[quat_start:]
                nonzero = np.flatnonzero(np.abs(quat) > 1e-10)
                if len(nonzero) and quat[nonzero[0]] < 0:
                    values[quat_start:] *= -1
            rounded = np.round(values, 4)
            rounded[rounded == 0] = 0  # Normalize signed zero before hashing.
            scene.append((name, rounded.tolist()))
        offset += width
        nv += velocities
    if len(state) != offset + nv:
        raise ValueError("XML joint layout does not match flattened simulator state")
    fixed = []
    for body in root.find("worldbody").findall("body"):
        name = body.get("name", "")
        if not name.startswith(("robot", "gripper")):
            fixed.append((name, body.get("pos", "0 0 0"), body.get("quat", "1 0 0 0")))
    return digest(json.dumps([scene, fixed], sort_keys=True).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Split directory is not empty; use a new directory to avoid changing frozen splits.")
    all_rows = []
    datasets = {}
    for task_id, task in enumerate(TASKS):
        path = args.data_dir / f"{task}_demo.hdf5"
        rows = []
        groups = {}
        with h5py.File(path, "r") as source:
            data = source["data"]
            for episode in sorted(data, key=lambda x: int(x.split("_")[-1])):
                demo = data[episode]
                state = demo["states"][0]
                group = scene_group(demo.attrs["model_file"], state)
                row = dict(task_id=task_id, task=task, dataset=path.name, episode=episode,
                           num_steps=len(demo["actions"]), initial_state_sha256=digest(state.tobytes()),
                           group_id=group)
                rows.append(row)
                groups.setdefault(group, []).append(row)
        if len(rows) != 50:
            raise ValueError(f"Expected 50 episodes for {task}; found {len(rows)}")
        targets = {"train": 35, "validation": 8 if task_id == 0 else 7,
                   "test": 7 if task_id == 0 else 8}
        counts = dict.fromkeys(targets, 0)
        group_list = list(groups.values())
        random.Random(args.seed + task_id).shuffle(group_list)
        # Inspected demos and any matching-scene episodes belong in training.
        group_list.sort(key=lambda group: not any(row["episode"] == "demo_0" for row in group))
        for group in group_list:
            if any(row["episode"] == "demo_0" for row in group):
                split = "train"
            else:
                fitting = [name for name in targets if counts[name] + len(group) <= targets[name]]
                candidates = fitting or list(targets)
                split = max(candidates, key=lambda name: (targets[name] - counts[name]) / targets[name])
            for row in group:
                row["split"] = split
            counts[split] += len(group)
        assert len({row["episode"] for row in rows}) == 50
        assert all(len({row["split"] for row in group}) == 1 for group in groups.values())
        with path.open("rb") as stream:
            file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        datasets[task] = dict(file=path.name, sha256=file_hash, episodes=len(rows),
                              scene_groups=len(groups), targets=targets, counts=counts)
        all_rows.extend(rows)
        print(f"{task}: {counts}; distinct starting-scene groups={len(groups)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0])
    for split in ("train", "validation", "test"):
        with (args.output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for row in all_rows if row["split"] == split)
    summary = dict(seed=args.seed, datasets=datasets,
                   totals={split: sum(row["split"] == split for row in all_rows)
                           for split in ("train", "validation", "test")},
                   grouping="Whole episode; matching non-robot joint positions rounded to 4 decimals and fixed scene-body poses stay together within each task. Robot state and simulator time are excluded from scene grouping. XML joint layout is checked against state length.",
                   limitations="No session identifiers were supplied. Session independence and near-matching scenes across rounding boundaries cannot be guaranteed.",
                   inspected_episodes="demo_0 of both tasks assigned to training; no model selection on test episodes.")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Total:", summary["totals"], flush=True)
    print("Manifests saved to", args.output_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
