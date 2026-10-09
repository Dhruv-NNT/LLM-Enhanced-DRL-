#!/usr/bin/env python3
"""Build a small, fast-loading TensorBoard logdir from large training event files.

The raw comparison logdir is ~2 GB / ~100 tags / ~30M scalar points. TensorBoard's
backend handles that fine, but the browser has to parse a ~54 MB JSON payload and
draw ~750k points, which is what makes the page hang.

This script rewrites the runs keeping only the tags you actually plot, dropping
duplicated steps left behind by training restarts, and downsampling each series.

Usage:
    python3 make_comparison_logdir.py --src tensorboard_comparison --dst tb_small
    python3 make_comparison_logdir.py --src tensorboard_comparison --dst tb_small \
        --include '^train/(success|collision)$' --include '^loss/' --max-points 500
"""

import argparse
import collections
import os
import re
import struct
import sys

from tensorboard.compat.proto import event_pb2

DEFAULT_INCLUDE = [
    r"^train/(success|collision|weather_failure|truncated)$",
    r"^train/(success|collision_or_weather|truncation)_rate_running$",
    r"^train/reward/(episode_total_return|policy_controlled_return)$",
    r"^train/shadow/",
    r"^train/llm/(positive_label_count|return_weight_hybrid_mean|action_match_rate)$",
    r"^eval/",
    r"^loss/(loss|policy_loss|value_loss|entropy|llm_loss|distill_loss)$",
]


def read_records(path):
    """Yield raw serialized Event payloads from a TFRecord-format event file."""
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                return
            (length,) = struct.unpack("<Q", header)
            f.read(4)  # header crc
            payload = f.read(length)
            f.read(4)  # payload crc
            if len(payload) < length:
                return  # truncated tail from a killed run
            yield payload


def event_files(run_dir):
    """Event files for one run, ordered by the writer timestamp in the filename."""
    files = []
    for name in os.listdir(run_dir):
        if "tfevents" not in name:
            continue
        path = os.path.join(run_dir, name)
        if os.path.getsize(path) < 512:
            continue  # stub file from an aborted launch: header only, no summaries
        parts = name.split(".")
        try:
            stamp = float(parts[parts.index("tfevents") + 1])
        except (ValueError, IndexError):
            stamp = os.path.getmtime(path)
        files.append((stamp, path))
    return [p for _, p in sorted(files)]


def collect_run(run_dir, matches):
    """Return {tag: [(step, value), ...]} with restart overlaps removed."""
    series = collections.defaultdict(list)
    last_step = {}
    probes = [t.encode() for t in ("scalars",)]  # cheap prefilter, see below
    del probes

    for path in event_files(run_dir):
        for payload in read_records(path):
            event = event_pb2.Event()
            try:
                event.ParseFromString(payload)
            except Exception:
                continue
            if not event.HasField("summary"):
                continue
            step = event.step
            for value in event.summary.value:
                tag = value.tag
                if not matches(tag):
                    continue
                # A restart rewinds the step counter and replays a stretch of
                # history. Keeping only strictly increasing steps per tag drops
                # the duplicated segment instead of drawing it twice.
                if tag in last_step and step <= last_step[tag]:
                    continue
                kind = value.WhichOneof("value")
                if kind == "simple_value":
                    scalar = value.simple_value
                elif kind == "tensor":
                    try:
                        scalar = struct.unpack("<f", value.tensor.tensor_content)[0]
                    except struct.error:
                        if value.tensor.float_val:
                            scalar = value.tensor.float_val[0]
                        elif value.tensor.double_val:
                            scalar = value.tensor.double_val[0]
                        else:
                            continue
                else:
                    continue
                series[tag].append((step, float(scalar)))
                last_step[tag] = step
    return series


def downsample(points, max_points):
    """Uniform stride, always keeping the first and last point."""
    if len(points) <= max_points:
        return points
    stride = len(points) / float(max_points)
    picked = [points[int(i * stride)] for i in range(max_points)]
    if picked[-1] != points[-1]:
        picked[-1] = points[-1]
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source logdir with the large runs")
    ap.add_argument("--dst", required=True, help="destination logdir to create")
    ap.add_argument("--include", action="append", default=None,
                    help="regex of tags to keep (repeatable; default: a curated set)")
    ap.add_argument("--max-points", type=int, default=1000,
                    help="max points kept per tag per run (default: 1000)")
    ap.add_argument("--list-tags", action="store_true",
                    help="print every tag found and exit without writing")
    args = ap.parse_args()

    patterns = [re.compile(p) for p in (args.include or DEFAULT_INCLUDE)]
    keep_all = args.list_tags
    matches = (lambda tag: True) if keep_all else (
        lambda tag: any(p.search(tag) for p in patterns))

    runs = sorted(d for d in os.listdir(args.src)
                  if os.path.isdir(os.path.join(args.src, d)))
    if not runs:
        sys.exit(f"no run subdirectories under {args.src}")

    if args.list_tags:
        found = set()
        for run in runs:
            found |= set(collect_run(os.path.join(args.src, run), matches))
        for tag in sorted(found):
            print(tag)
        return

    from torch.utils.tensorboard import SummaryWriter

    total_in = total_out = 0
    axis_report = collections.defaultdict(set)

    for run in runs:
        src_run = os.path.join(args.src, run)
        series = collect_run(src_run, matches)
        if not series:
            print(f"{run:22s} no matching tags, skipped")
            continue

        out_dir = os.path.join(args.dst, run)
        writer = SummaryWriter(log_dir=out_dir)
        kept = 0
        for tag, points in sorted(series.items()):
            total_in += len(points)
            # Two step axes are in use: episode-indexed tags and agent_steps
            # tags. Record which one each tag lives on so the mismatch is visible.
            axis_report["episode" if points[-1][0] < 200000 else "agent_steps"].add(tag)
            for step, value in downsample(points, args.max_points):
                writer.add_scalar(tag, value, step)
                kept += 1
        writer.close()
        total_out += kept
        print(f"{run:22s} {len(series):3d} tags  {sum(len(v) for v in series.values()):>9,} -> {kept:>7,} points")

    src_bytes = sum(os.path.getsize(os.path.join(dp, f))
                    for dp, _, fs in os.walk(args.src) for f in fs)
    dst_bytes = sum(os.path.getsize(os.path.join(dp, f))
                    for dp, _, fs in os.walk(args.dst) for f in fs)
    print(f"\n{total_in:,} -> {total_out:,} points   "
          f"{src_bytes/1e6:.0f} MB -> {dst_bytes/1e6:.1f} MB")

    if len(axis_report) > 1:
        print("\nWarning: tags use two different x axes, so they are not directly "
              "comparable on one chart:")
        for axis, tags in sorted(axis_report.items()):
            sample = ", ".join(sorted(tags)[:3])
            print(f"  {axis:12s} {len(tags):3d} tags  e.g. {sample}")

    print(f"\n  tensorboard --logdir {args.dst} --samples_per_plugin scalars=1000")


if __name__ == "__main__":
    main()
