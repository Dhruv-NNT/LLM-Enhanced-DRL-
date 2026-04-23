#!/usr/bin/env python3
"""
Extract final total reward values from memory-vs-no-memory episode frames.

The environment writes the total reward into the rendered plot title, not into
the GIF metadata. This script uses each episode's final PNG frame when present,
crops the title area, asks an Ollama vision model to read the number after
"Total Reward", and saves one consolidated JSON file. GIF decoding is kept only
as a fallback for episode folders that do not have saved frames.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
from PIL import Image, ImageSequence
from PIL import ImageEnhance, ImageFilter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SETTINGS = ("no_memory", "with_memory")
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
DECIMAL_RE = re.compile(r"[-+]?\d+\.\d+")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_root",
        type=Path,
        help="Path to a memory_vs_no_memory_<timestamp> output folder.",
    )
    parser.add_argument(
        "--model",
        default="gemma3:12b",
        help="Ollama vision model to use, for example gemma3:12b, llava:latest, or llama3.2-vision:latest.",
    )
    parser.add_argument("--setting", choices=SETTINGS, default=None)
    parser.add_argument("--episode", default=None, help="Optional episode id filter, for example ep001.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of episodes to process.")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Where to write the consolidated JSON. Defaults to <output_root>/total_rewards.json.",
    )
    parser.add_argument(
        "--sum-chart",
        type=Path,
        default=None,
        help="Where to write the total reward sum chart. Defaults to <output_root>/total_reward_sums.png.",
    )
    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=None,
        help="Where to save extracted title crops. Defaults to <output_root>/reward_extraction/title_crops.",
    )
    parser.add_argument(
        "--crop-height",
        type=int,
        default=120,
        help="Number of pixels to keep from the top of the last frame.",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=4,
        help="Upscale factor for the title crop sent to the vision model.",
    )
    return parser.parse_args()


def _episode_sources(
    output_root: Path,
    *,
    setting_filter: Optional[str] = None,
    episode_filter: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[tuple[str, str, Path, str, Optional[Path]]]:
    sources: List[tuple[str, str, Path, str, Optional[Path]]] = []
    for setting in SETTINGS:
        if setting_filter is not None and setting != setting_filter:
            continue
        setting_dir = output_root / setting
        if not setting_dir.exists():
            continue
        for episode_dir in sorted(setting_dir.glob("ep*")):
            if not episode_dir.is_dir():
                continue
            if episode_filter is not None and episode_dir.name != episode_filter:
                continue
            gif_path = episode_dir / f"{episode_dir.name}.gif"
            frame_paths = sorted((episode_dir / "frames").glob("image_*.png"))
            if frame_paths:
                sources.append((setting, episode_dir.name, frame_paths[-1], "frame", gif_path if gif_path.exists() else None))
            elif gif_path.exists():
                sources.append((setting, episode_dir.name, gif_path, "gif", gif_path))
            if limit is not None and len(sources) >= int(limit):
                return sources
    return sources


def _load_source_image(source_path: Path, source_type: str) -> Image.Image:
    if source_type == "frame":
        return Image.open(source_path).convert("RGB")

    with Image.open(source_path) as image:
        last_frame: Optional[Image.Image] = None
        for frame in ImageSequence.Iterator(image):
            last_frame = frame.convert("RGB")
        if last_frame is None:
            raise RuntimeError(f"No frames found in {source_path}")
        return last_frame


def _extract_title_crop(
    source_path: Path,
    source_type: str,
    crop_path: Path,
    crop_height: int,
    upscale: int,
) -> None:
    source_image = _load_source_image(source_path, source_type)
    width, height = source_image.size
    crop_bottom = min(max(1, int(crop_height)), height)
    # Focus on the middle of the title where "Total Reward <value>" appears.
    left = int(width * 0.24)
    right = int(width * 0.82)
    title_crop = source_image.crop((left, 0, right, crop_bottom))

    # Make the thin plot title easier to read, especially the minus sign and decimal point.
    scale = max(1, int(upscale))
    title_crop = title_crop.resize((title_crop.width * scale, title_crop.height * scale), Image.Resampling.LANCZOS)
    title_crop = title_crop.convert("L")
    title_crop = ImageEnhance.Contrast(title_crop).enhance(1.8)
    title_crop = ImageEnhance.Sharpness(title_crop).enhance(2.0)
    title_crop = title_crop.filter(ImageFilter.SHARPEN)
    title_crop = title_crop.convert("RGB")
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    title_crop.save(crop_path)


def _extract_json_object(text: str) -> Dict[str, Any]:
    match = JSON_OBJECT_RE.search(text.strip())
    if not match:
        raise ValueError("No JSON object found in model response.")
    return json.loads(match.group(0))


def _parse_total_reward(raw_text: str) -> float:
    try:
        payload = _extract_json_object(raw_text)
        value = payload.get("total_reward_text")
        if value is None:
            value = payload.get("total_reward")
        if value is not None:
            match = DECIMAL_RE.search(str(value))
            if match:
                return float(match.group(0))
    except Exception:
        pass

    lowered = raw_text.lower()
    idx = lowered.find("total_reward")
    if idx < 0:
        idx = lowered.find("total reward")
    search_text = raw_text[idx:] if idx >= 0 else raw_text
    match = DECIMAL_RE.search(search_text)
    if not match:
        raise ValueError(
            "Could not parse a decimal total reward from model response. "
            f"The model may have dropped the decimal point: {raw_text!r}"
        )
    return float(match.group(0))


def _ask_ollama_for_total_reward(*, model: str, image_path: Path) -> tuple[float, str]:
    import ollama

    prompt = (
        "Read only the plot title text in this image.\n"
        "Find the exact keyword 'Total Reward' or 'Total Rewards'.\n"
        "Extract the numeric token immediately after that keyword.\n"
        "The value may be negative. If it starts with '-', you must keep the minus sign.\n"
        "The value may contain a decimal point. You must keep the decimal point and all visible digits.\n"
        "Do not use the number after 'Reward' unless it is the number after 'Total Reward'.\n"
        "Do not round. Do not infer a positive value if a minus sign is visible.\n\n"
        "Examples:\n"
        "- If the title says 'Total Reward -12.345', return -12.345.\n"
        "- If the title says 'Total Reward 1.360', return 1.360.\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\"total_reward_text\": \"<exact text after Total Reward>\", \"total_reward\": number}\n"
        "Do not include any explanation."
    )
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
        options={"temperature": 0.0},
    )
    raw_text = (response.get("message") or {}).get("content", "") or ""
    return _parse_total_reward(raw_text), raw_text


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for setting in SETTINGS:
        values = [
            float(record["total_reward"])
            for record in records
            if record["setting"] == setting and record.get("total_reward") is not None
        ]
        summary[setting] = {
            "episodes_extracted": len(values),
            "total_reward_sum": None if not values else sum(values),
            "min_total_reward": None if not values else min(values),
            "max_total_reward": None if not values else max(values),
        }
    return summary


def _plot_reward_sums(summary: Dict[str, Any], output_path: Path) -> None:
    labels = ["No Memory", "With Memory"]
    values = [
        summary["no_memory"]["total_reward_sum"],
        summary["with_memory"]["total_reward_sum"],
    ]
    numeric_values = [0.0 if value is None else float(value) for value in values]

    fig, axis = plt.subplots(figsize=(6.5, 5.0))
    colors = ["#1565c0", "#ef6c00"]
    bars = axis.bar(labels, numeric_values, color=colors, width=0.55)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Total Reward Sum")
    axis.set_ylabel("Sum of Total Rewards")
    axis.legend(bars, labels, frameon=False)
    for bar, value in zip(bars, values):
        if value is None:
            continue
        y = bar.get_height()
        offset = 3 if y >= 0 else -12
        va = "bottom" if y >= 0 else "top"
        axis.annotate(
            f"{float(value):.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    output_root = args.output_root.resolve()
    if not output_root.exists():
        raise FileNotFoundError(output_root)

    out_json = args.out_json or (output_root / "total_rewards.json")
    sum_chart = args.sum_chart or (output_root / "total_reward_sums.png")
    frame_dir = args.frame_dir or (output_root / "reward_extraction" / "title_crops")

    episode_sources = _episode_sources(
        output_root,
        setting_filter=args.setting,
        episode_filter=args.episode,
        limit=args.limit,
    )
    if not episode_sources:
        raise RuntimeError(f"No episode frames or GIFs found under {output_root}")

    records: List[Dict[str, Any]] = []
    for setting, episode_id, source_path, source_type, gif_path in episode_sources:
        crop_path = frame_dir / setting / f"{episode_id}_title.png"
        record: Dict[str, Any] = {
            "setting": setting,
            "episode_id": episode_id,
            "source_type": source_type,
            "source_path": str(source_path),
            "gif_path": None if gif_path is None else str(gif_path),
            "title_crop_path": str(crop_path),
            "total_reward": None,
            "status": "pending",
            "error": None,
            "ollama_model": str(args.model),
            "ollama_response": "",
        }
        try:
            _extract_title_crop(
                source_path,
                source_type,
                crop_path,
                int(args.crop_height),
                int(args.upscale),
            )
            total_reward, raw_response = _ask_ollama_for_total_reward(model=str(args.model), image_path=crop_path)
            record["total_reward"] = float(total_reward)
            record["status"] = "ok"
            record["ollama_response"] = raw_response
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        print(
            f"[reward-extract] {setting}/{episode_id} status={record['status']} "
            f"total_reward={record['total_reward']}",
            flush=True,
        )

    summary = _summarize(records)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "output_root": str(output_root),
        "ollama_model": str(args.model),
        "source": "final episode frame title OCR using Ollama vision model",
        "records": records,
        "summary": summary,
        "charts": {
            "sum_chart": str(sum_chart),
        },
    }
    _plot_reward_sums(summary, sum_chart)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved consolidated rewards JSON -> {out_json}", flush=True)
    print(f"Saved total reward sum chart -> {sum_chart}", flush=True)


if __name__ == "__main__":
    main()
