"""Experiment A: paired (flight-by-flight) analysis of existing evaluation JSONs.

All models were evaluated on the SAME scenario seeds, so outcomes are paired. This script
reads eval_*.json files (from evaluate_rl.py or evaluate_per_aircraft.py) and produces:
  T1  each run vs a baseline: success %, delta (pp), rescued/lost counts, exact McNemar p,
      paired-bootstrap 95% CI on the delta
  T2  failure-mode split per run (collision / weather / truncation %)
  T3  pairwise complementarity: for every pair of runs, McNemar tests on the collision
      indicator, the weather indicator, and success
  T4  rescue analysis on the scenarios the baseline failed
No scipy needed. Writes a markdown report and prints it. Post-processing only.

Example:
    python new_approach/analyze_pairwise.py \
        --pattern "Evaluation_distillation_approach/*/eval_last.json" \
        --baseline T0_baseline_seed0 --out Evaluation_distillation_approach/analysis_A_final5M.md
"""
from __future__ import annotations
import argparse, glob, json, math, os, random
from itertools import combinations
from typing import Dict, List


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load_runs(pattern: str, runs_filter: List[str] | None) -> Dict[str, Dict[int, dict]]:
    out = {}
    for f in sorted(glob.glob(pattern)):
        name = os.path.basename(os.path.dirname(f))
        if runs_filter and name not in runs_filter:
            continue
        d = json.load(open(f))
        out[name] = {int(e["seed"]): e for e in d["episodes"]}
    return out


def boot_ci(seeds, A, B, n_boot, rng):
    diffs = []
    n = len(seeds)
    for _ in range(n_boot):
        ss = [seeds[rng.randrange(n)] for _ in range(n)]
        diffs.append(100.0 * (sum(B[s]["success"] for s in ss) - sum(A[s]["success"] for s in ss)) / n)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="Evaluation_distillation_approach/*/eval_last.json")
    ap.add_argument("--baseline", default="T0_baseline_seed0")
    ap.add_argument("--runs", nargs="*", default=None, help="optional subset of run names")
    ap.add_argument("--out", default="Evaluation_distillation_approach/analysis_pairwise.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.boot_seed)

    runs = load_runs(args.pattern, args.runs)
    if args.baseline not in runs:
        raise SystemExit(f"baseline {args.baseline} not found among {list(runs)}")
    base = runs[args.baseline]
    common = sorted(set.intersection(*[set(r) for r in runs.values()]))
    L: List[str] = []
    L.append(f"# Paired analysis — pattern `{args.pattern}`\n")
    L.append(f"Baseline: **{args.baseline}**. Runs: {len(runs)}. Common paired scenarios: **{len(common)}**.\n")

    # T1 ---------------------------------------------------------------
    L.append("## T1. Each run vs baseline (paired on identical scenarios)\n")
    L.append("| run | success % | Δ pp | rescued | lost | McNemar p | 95% CI Δpp | real? |")
    L.append("|---|---|---|---|---|---|---|---|")
    sb = 100 * sum(base[s]["success"] for s in common) / len(common)
    L.append(f"| {args.baseline} (baseline) | {sb:.1f} | 0.0 | – | – | – | – | – |")
    for name, M in sorted(runs.items(), key=lambda kv: -sum(kv[1][s]['success'] for s in common)):
        if name == args.baseline:
            continue
        b = sum(1 for s in common if (not base[s]["success"]) and M[s]["success"])
        c = sum(1 for s in common if base[s]["success"] and (not M[s]["success"]))
        sm = 100 * sum(M[s]["success"] for s in common) / len(common)
        p = mcnemar_p(b, c)
        lo, hi = boot_ci(common, base, M, args.n_boot, rng)
        L.append(f"| {name} | {sm:.1f} | {sm - sb:+.1f} | {b} | {c} | {p:.3f} | [{lo:+.1f}, {hi:+.1f}] | {'YES' if p < 0.05 else 'no'} |")
    L.append("\n*rescued* = baseline failed, run succeeded; *lost* = the reverse. "
             "*real?* = p < 0.05 (the split of disagreements is too lopsided to be a coin flip).\n")

    # T2 ---------------------------------------------------------------
    L.append("## T2. Failure-mode split (% of scenarios)\n")
    L.append("| run | success | collision | weather | truncated |")
    L.append("|---|---|---|---|---|")
    for name, M in sorted(runs.items(), key=lambda kv: -sum(kv[1][s]['success'] for s in common)):
        n = len(common)
        col = 100 * sum(M[s]["failure_reason"] == "collision" for s in common) / n
        wx = 100 * sum(M[s]["failure_reason"] == "weather" for s in common) / n
        tr = 100 * sum(M[s]["truncated"] for s in common) / n
        su = 100 * sum(M[s]["success"] for s in common) / n
        L.append(f"| {name} | {su:.1f} | {col:.1f} | {wx:.1f} | {tr:.1f} |")
    L.append("")

    # T3 ---------------------------------------------------------------
    L.append("## T3. Pairwise complementarity (McNemar on each failure type; counts are 'A-only' vs 'B-only')\n")
    L.append("| A | B | collision A-only / B-only (p) | weather A-only / B-only (p) | success A-only / B-only (p) |")
    L.append("|---|---|---|---|---|")
    names = sorted(runs)
    for a, bname in combinations(names, 2):
        A, B = runs[a], runs[bname]
        cells = []
        for mode in ("collision", "weather"):
            x = sum(1 for s in common if A[s]["failure_reason"] == mode and B[s]["failure_reason"] != mode)
            y = sum(1 for s in common if B[s]["failure_reason"] == mode and A[s]["failure_reason"] != mode)
            p = mcnemar_p(x, y)
            cells.append(f"{x} / {y} ({p:.3f}){' *' if p < 0.05 else ''}")
        x = sum(1 for s in common if A[s]["success"] and not B[s]["success"])
        y = sum(1 for s in common if B[s]["success"] and not A[s]["success"])
        p = mcnemar_p(x, y)
        cells.append(f"{x} / {y} ({p:.3f}){' *' if p < 0.05 else ''}")
        L.append(f"| {a} | {bname} | " + " | ".join(cells) + " |")
    L.append("\n`*` = p < 0.05. A lower 'A-only' count means A makes that mistake less often than B.\n")

    # T4 ---------------------------------------------------------------
    failed = [s for s in common if not base[s]["success"]]
    L.append(f"## T4. Rescue analysis on the {len(failed)} scenarios the baseline failed\n")
    L.append("| run | rescued | % of baseline failures |")
    L.append("|---|---|---|")
    for name, M in sorted(runs.items(), key=lambda kv: -sum(kv[1][s]['success'] for s in failed)):
        if name == args.baseline:
            continue
        r = sum(M[s]["success"] for s in failed)
        L.append(f"| {name} | {r} | {100 * r / max(1, len(failed)):.1f} |")
    text = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write(text)
    print(text)
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
