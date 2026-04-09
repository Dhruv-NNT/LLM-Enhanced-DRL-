#!/usr/bin/env python3
"""Analyze pure-LLM run logs and summarize guidance behavior."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Optional


def wrap_deg(d: float) -> float:
    return ((d + 180.0) % 360.0) - 180.0


def load_index(index_path: Path) -> List[dict]:
    rows = []
    if not index_path.exists():
        return rows
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def analyze_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    memory_dir = Path(meta["memory_dir"])
    index_path = Path(meta["memory_index"])
    rows = load_index(index_path)

    run_id = None
    if memory_dir.name.startswith("run_"):
        run_id = memory_dir.name
    if run_id:
        rows = [r for r in rows if r.get("run_id") == run_id or str(r.get("mem_path", "")).startswith(str(memory_dir))]

    phases = Counter(r.get("phase") for r in rows)
    turns = [r.get("llm_turn_deg", r.get("turn_deg")) for r in rows]
    turns = [t for t in turns if isinstance(t, (int, float))]
    turn_hist = Counter(int(t) for t in turns)
    minimal = sum(1 for t in turns if abs(t) <= 5)

    empty_tech = sum(1 for r in rows if not (r.get("tech_summary_snip") or "").strip())
    empty_rat = sum(1 for r in rows if not (r.get("rationale_snip") or "").strip())

    steps = [r.get("step") for r in rows if isinstance(r.get("step"), int)]

    mem_files = sorted(memory_dir.glob("t*.json"))
    applied_counts = Counter()
    applied_examples = []
    for mf in mem_files:
        try:
            rec = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        man = (rec.get("answer") or {}).get("maneuver") or {}
        turn = man.get("heading_change_deg", 0)
        before = (rec.get("observation_before") or {}).get("own_heading_deg")
        after = (rec.get("observation_after") or {}).get("own_heading_deg")
        if before is None or after is None:
            continue
        delta = wrap_deg(float(after) - float(before))
        if abs(delta - float(turn)) <= 2.0:
            applied = "applied"
        elif abs(float(turn)) > 0 and abs(delta) <= 2.0:
            applied = "not_applied"
        else:
            applied = "mismatch"
        applied_counts[applied] += 1
        if applied != "applied" and len(applied_examples) < 3:
            applied_examples.append({
                "file": str(mf),
                "turn": float(turn),
                "before": float(before),
                "after": float(after),
                "delta": float(delta),
                "class": applied,
            })

    return {
        "meta": meta,
        "count": len(rows),
        "phases": phases,
        "turn_hist": turn_hist,
        "minimal_pct": (minimal / len(turns) * 100.0) if turns else 0.0,
        "empty_tech_pct": (empty_tech / len(rows) * 100.0) if rows else 0.0,
        "empty_rat_pct": (empty_rat / len(rows) * 100.0) if rows else 0.0,
        "steps": steps,
        "applied_counts": applied_counts,
        "applied_examples": applied_examples,
        "memory_files_count": len(mem_files),
        "memory_dir": str(memory_dir),
        "index_path": str(index_path),
    }


def _print_stats(label: str, stats: dict) -> None:
    print(label)
    print("  calls:", stats["count"], "phases:", dict(stats["phases"]))
    print("  turn_hist:", dict(stats["turn_hist"]))
    print("  minimal %:", f"{stats['minimal_pct']:.1f}%")
    print("  empty tech %:", f"{stats['empty_tech_pct']:.1f}%", "empty rationale %:", f"{stats['empty_rat_pct']:.1f}%")
    steps = stats["steps"]
    print("  steps: min", min(steps) if steps else None, "max", max(steps) if steps else None, "unique", len(set(steps)))
    print("  mem files:", stats["memory_files_count"], "applied counts:", dict(stats["applied_counts"]))
    if stats["applied_examples"]:
        print("  applied examples:")
        for ex in stats["applied_examples"]:
            print("   ", ex)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze pure-LLM run logs.")
    parser.add_argument("--no-run", required=True, help="Path to no_memory run directory.")
    parser.add_argument("--with-run", required=True, help="Path to with_memory run directory.")
    args = parser.parse_args()

    no_stats = analyze_run(Path(args.no_run))
    with_stats = analyze_run(Path(args.with_run))

    _print_stats("NO_MEMORY", no_stats)
    print()
    _print_stats("WITH_MEMORY", with_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
