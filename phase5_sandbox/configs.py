"""Defaults for the 4-phase sandbox evaluation harness."""

from pathlib import Path

DEFAULT_EPISODES = 10
BASE_SEED = 12345
FPS = 5.0
SAFE_R = 5.0
KEEP_FRAMES = False

OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"
