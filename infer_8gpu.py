"""
8-GPU data-parallel inference for LTX-2.
Each GPU runs an independent inference process on its share of CSV rows.

CSV format:
  caption,keyframes
  "prompt text",frame_idx:/path/to/img;frame_idx:/path/to/img;...

Usage:
  python infer_8gpu.py --csv /path/to/tasks.csv --num-gpus 8 [other args]
"""

import argparse
import csv
import os
import subprocess
import sys
from multiprocessing import Process


def parse_args():
    parser = argparse.ArgumentParser(description="8-GPU DP inference for LTX-2")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with tasks")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")

    # Fixed model paths
    parser.add_argument("--checkpoint-path", type=str,
                        default="/yke/models/LTX-2.3/ltx-2.3-22b-dev.safetensors")
    parser.add_argument("--distilled-lora", type=str,
                        default="/yke/models/LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors")
    parser.add_argument("--distilled-lora-strength", type=float, default=0.8)
    parser.add_argument("--lora", type=str, default=None,
                        help="Optional LoRA path")
    parser.add_argument("--spatial-upsampler-path", type=str,
                        default="/yke/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
    parser.add_argument("--gemma-root", type=str,
                        default="/yke/models/gemma-3-12b-it-qat-q4_0-unquantized")

    # Generation params
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--frame-rate", type=int, default=24)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-strength", type=float, default=1.0,
                        help="Default strength for keyframe images")
    parser.add_argument("--first-frame", type=str, default=None,
                        help="Optional first frame image (frame 0) applied to all tasks")

    return parser.parse_args()


def read_csv(csv_path: str) -> list[dict]:
    """Read CSV and return list of task dicts."""
    tasks = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            caption = row["caption"].strip()
            keyframes_raw = row.get("keyframes", "").strip()

            # Parse keyframes: "144:/path;168:/path" -> [(144, "/path"), ...]
            keyframes = []
            if keyframes_raw:
                for part in keyframes_raw.split(";"):
                    part = part.strip()
                    if not part:
                        continue
                    frame_idx_str, img_path = part.split(":", 1)
                    keyframes.append((int(frame_idx_str), img_path.strip()))

            tasks.append({
                "idx": idx,
                "caption": caption,
                "keyframes": keyframes,
            })
    return tasks


def build_command(task: dict, args: argparse.Namespace) -> list[str]:
    """Build the inference command for a single task."""
    output_path = os.path.join(args.output_dir, f"{task['idx']:06d}.mp4")

    cmd = [
        sys.executable, "-m", "ltx_pipelines.ti2vid_two_stages",
        "--checkpoint-path", args.checkpoint_path,
        "--distilled-lora", args.distilled_lora, str(args.distilled_lora_strength),
        "--spatial-upsampler-path", args.spatial_upsampler_path,
        "--gemma-root", args.gemma_root,
        "--prompt", task["caption"],
        "--output-path", output_path,
        "--num-inference-steps", str(args.num_inference_steps),
        "--height", str(args.height),
        "--width", str(args.width),
        "--frame-rate", str(args.frame_rate),
        "--num-frames", str(args.num_frames),
        "--seed", str(args.seed),
    ]

    if args.lora:
        cmd.extend(["--lora", args.lora])

    # Add first frame if specified
    if args.first_frame:
        cmd.extend(["--image", args.first_frame, "0", str(args.image_strength)])

    # Add keyframes from CSV
    for frame_idx, img_path in task["keyframes"]:
        cmd.extend(["--image", img_path, str(frame_idx), str(args.image_strength)])

    return cmd


def worker(gpu_id: int, tasks: list[dict], args: argparse.Namespace):
    """Worker process: run assigned tasks sequentially on one GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    for task in tasks:
        output_path = os.path.join(args.output_dir, f"{task['idx']:06d}.mp4")
        if os.path.exists(output_path):
            print(f"[GPU {gpu_id}] Skipping task {task['idx']}, output already exists: {output_path}")
            continue

        cmd = build_command(task, args)
        print(f"[GPU {gpu_id}] Starting task {task['idx']} -> {output_path}")
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"[GPU {gpu_id}] Finished task {task['idx']}")
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] FAILED task {task['idx']}: exit code {e.returncode}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = read_csv(args.csv)
    print(f"Loaded {len(tasks)} tasks from {args.csv}")

    num_gpus = min(args.num_gpus, len(tasks))
    print(f"Using {num_gpus} GPUs")

    # Round-robin distribute tasks to GPUs
    gpu_tasks: list[list[dict]] = [[] for _ in range(num_gpus)]
    for i, task in enumerate(tasks):
        gpu_tasks[i % num_gpus].append(task)

    for gpu_id in range(num_gpus):
        print(f"  GPU {gpu_id}: {len(gpu_tasks[gpu_id])} tasks")

    # Launch workers
    processes = []
    for gpu_id in range(num_gpus):
        if not gpu_tasks[gpu_id]:
            continue
        p = Process(target=worker, args=(gpu_id, gpu_tasks[gpu_id], args))
        p.start()
        processes.append(p)

    # Wait for all workers
    for p in processes:
        p.join()

    print("All done.")


if __name__ == "__main__":
    main()
