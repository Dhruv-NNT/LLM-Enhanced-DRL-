"""Dummy test for sending one prompt plus four images to Ollama.

This mirrors the controller's vision request shape:

message = {
    "role": "user",
    "content": prompt_text,
    "images": [base64_png_0, base64_png_1, base64_png_2, base64_png_3],
}
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

from configs import OLLAMA_HOST, OLLAMA_MODEL

DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent / "outputs" / "global_text_test_6"
DEFAULT_IMAGE_NAMES = ("image_000.png", "image_001.png", "image_002.png", "image_003.png")


def _make_dummy_images(image_dir: Path) -> list[Path]:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to generate dummy PNG images.") from exc

    image_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#e0f2fe", "#dcfce7", "#fef9c3", "#fee2e2"]
    labels = [
        "SNAPSHOT 0 CURRENT",
        "SNAPSHOT 1 PREVIOUS",
        "SNAPSHOT 2 PREVIOUS",
        "SNAPSHOT 3 OLDEST",
    ]
    paths: list[Path] = []
    for idx, (color, label) in enumerate(zip(colors, labels)):
        img = Image.new("RGB", (420, 220), color)
        draw = ImageDraw.Draw(img)
        draw.rectangle((20, 20, 400, 200), outline="#111827", width=4)
        draw.text((42, 84), label, fill="#111827")
        draw.text((42, 124), f"dummy frame index={idx}", fill="#111827")
        path = image_dir / f"dummy_snapshot_{idx}.png"
        img.save(path)
        paths.append(path)
    return paths


def _encode_images(image_paths: Sequence[Path]) -> list[str]:
    encoded: list[str] = []
    for image_path in image_paths:
        with image_path.open("rb") as handle:
            encoded.append(base64.b64encode(handle.read()).decode("ascii"))
    return encoded


def _recent_frame_paths(latest_frame_path: Path, count: int = 4) -> list[Path]:
    import re

    match = re.match(r"image_(\d+)\.png$", latest_frame_path.name)
    if not match:
        return [latest_frame_path] if latest_frame_path.exists() else []
    step = int(match.group(1))
    paths: list[Path] = []
    for value in range(step, max(-1, step - int(count)), -1):
        candidate = latest_frame_path.with_name(f"image_{value:03d}.png")
        if candidate.exists():
            paths.append(candidate)
    return paths


def _resolve_image_paths(args: argparse.Namespace, temp_dir: Path) -> list[Path]:
    if args.image:
        paths = [Path(value).expanduser().resolve() for value in args.image]
    elif args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        if args.latest_frame:
            latest = Path(args.latest_frame).expanduser()
            latest_path = latest.resolve() if latest.is_absolute() else (output_dir / latest).resolve()
        else:
            candidates = sorted(output_dir.glob("image_*.png"))
            latest_path = candidates[-1] if candidates else output_dir / "image_000.png"
        paths = _recent_frame_paths(latest_path, count=int(args.count))
    elif args.use_dummy_images:
        paths = _make_dummy_images(temp_dir)
    else:
        paths = [DEFAULT_IMAGE_DIR / name for name in DEFAULT_IMAGE_NAMES]

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing image file(s): " + ", ".join(missing))
    if not paths:
        raise FileNotFoundError("No image files found to send.")
    return paths[: int(args.count)]


def _build_payload(
    model: str,
    prompt_text: str,
    image_b64: Sequence[str],
    *,
    num_predict: int,
) -> dict[str, object]:
    message: dict[str, object] = {"role": "user", "content": prompt_text}
    if image_b64:
        message["images"] = list(image_b64)
    return {
        "model": model,
        "stream": False,
        "messages": [message],
        "options": {
            "temperature": 0,
            "num_predict": int(num_predict),
        },
    }


def _post_to_ollama(host: str, payload: dict[str, object], timeout: float) -> str:
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw_body = response.read().decode("utf-8")
    parsed = json.loads(raw_body)
    return str((parsed.get("message") or {}).get("content", raw_body))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", OLLAMA_HOST))
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true", help="Build and summarize payload without calling Ollama.")
    parser.add_argument("--keep-images", action="store_true", help="Keep generated dummy PNG files and print their folder.")
    parser.add_argument("--image", action="append", help="Image path to send. Repeat up to four times; order is preserved.")
    parser.add_argument("--output-dir", help="Folder containing image_000.png, image_001.png, etc.")
    parser.add_argument("--latest-frame", help="Latest frame name/path. With --output-dir, sends this frame plus previous frames.")
    parser.add_argument("--count", type=int, default=4, help="Maximum number of images to send.")
    parser.add_argument("--prompt", default=None, help="Override the test prompt text.")
    parser.add_argument("--use-dummy-images", action="store_true", help="Generate temporary labeled dummy images instead of using default output frames.")
    parser.add_argument("--num-predict", type=int, default=1024, help="Maximum response tokens from Ollama.")
    args = parser.parse_args()

    prompt_text = args.prompt or (
        "You are testing multimodal aircraft snapshots. You should receive exactly four separate PNG images in order. "
        "Analyze all four images, not just the first two. Return four short numbered lines: "
        "Image 0, Image 1, Image 2, Image 3. For each line mention visible aircraft labels, weather rings, "
        "and any obvious change from the previous image. Then add one final sentence saying whether the text prompt "
        "and all four images arrived together."
    )

    with tempfile.TemporaryDirectory(prefix="ollama_vision_dummy_") as tmp:
        image_dir = Path(tmp)
        image_paths = _resolve_image_paths(args, image_dir)
        image_b64 = _encode_images(image_paths)
        payload = _build_payload(args.model, prompt_text, image_b64, num_predict=int(args.num_predict))
        message = payload["messages"][0]  # type: ignore[index]

        print("payload_summary:")
        print(f"- model={args.model}")
        print(f"- host={args.host}")
        print(f"- content_chars={len(prompt_text)}")
        print(f"- num_predict={int(args.num_predict)}")
        print(f"- image_count={len(image_b64)}")
        print(f"- message_keys={sorted(message.keys())}")  # type: ignore[union-attr]
        print(f"- image_paths={[str(path) for path in image_paths]}")

        if args.keep_images:
            kept_dir = Path.cwd() / "multi-agent-isolated-llm-guidance" / "outputs" / "dummy_vision_payload"
            kept_dir.mkdir(parents=True, exist_ok=True)
            for path in image_paths:
                (kept_dir / path.name).write_bytes(path.read_bytes())
            print(f"- kept_images_dir={kept_dir}")

        if args.dry_run:
            print("dry_run_ok=true")
            return 0

        try:
            response_text = _post_to_ollama(args.host, payload, timeout=float(args.timeout))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            print(f"ollama_http_error={exc}")
            print(body)
            return 2
        except Exception as exc:
            print(f"ollama_error={type(exc).__name__}: {exc}")
            return 2

        print("ollama_response:")
        print(response_text)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
