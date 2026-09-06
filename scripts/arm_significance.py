#!/usr/bin/env python3
"""Is the Stage-D arm ranking real, or is it noise?

`val_resolved` is the mean of the per-data-source solve rates on the validation split
(SWE 112 rows + unit-test 400 rows, greedy n=1). The autoresearch loop keeps the arm with the
highest value, but it compares point estimates with no uncertainty attached. This script attaches
it, using the fact that every arm is evaluated on the SAME validation instances: a paired
comparison is far more sensitive than comparing two marginal rates.

For each arm vs the baseline, at the final validation step:
  * per-source rates and the noise floor (binomial SE of the point estimate)
  * McNemar exact test on the discordant pairs (binary metrics only)
  * stratified paired bootstrap CI on the difference

--metric picks what to rank on. Only measurements that do not depend on the reward config are
comparable across arms: answer_match (solve), f2p_frac, patch_apply_score, p2p_frac. `final_reward`
is NOT comparable -- each arm computes it with a different reward function.

Usage:
  python scripts/arm_significance.py [--metric answer_match] [--exp-glob 'experiments/exp_Q*']
                                     [--baseline exp_Q000_baseline]
                                     [--val data/curated/curated_v3/validation.parquet]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# metrics that measure the policy, not the reward function that trained it
COMPARABLE = ("answer_match", "f2p_frac", "patch_apply_score", "p2p_frac", "patch_format_score",
              "rule_correctness_score", "gated_out")


def val_instance_ids(val_parquet: str) -> set[str]:
    df = pd.read_parquet(val_parquet)
    return set(df["reward_model"].map(lambda x: x["ground_truth"]))


def load_arm(exp_dir: str, vids: set[str], metric: str) -> tuple[int, dict[str, tuple[str, float]]]:
    """-> (final validation step, {instance_id: (track, resolved 0/1)}) for that step."""
    path = os.path.join(exp_dir, "instance_log.jsonl")
    if not os.path.exists(path):
        return -1, {}
    by_step: dict[int, dict[str, tuple[str, float]]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:      # a run killed mid-write leaves a partial last line
                continue
            iid = r.get("instance_id")
            if iid not in vids:
                continue
            if r.get("infra_excluded"):       # infra failure, not a policy failure
                continue
            by_step[int(r.get("step", -1))][iid] = (r.get("track", "?"), float(r.get(metric, 0.0)))
    if not by_step:
        return -1, {}
    step = max(by_step)
    return step, by_step[step]


def resolved(scores: dict[str, tuple[str, float]]) -> tuple[float, dict[str, tuple[float, int]]]:
    """Unweighted mean of the per-track means (the shape val_resolved has)."""
    per = defaultdict(list)
    for track, ok in scores.values():
        per[track].append(ok)
    rates = {t: (sum(v) / len(v), len(v)) for t, v in per.items()}
    return (sum(r for r, _ in rates.values()) / len(rates) if rates else float("nan")), rates


def binom_se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs (b wins for A, c wins for B)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap(pairs: dict[str, list[tuple[float, float]]], iters: int, seed: int) -> tuple[float, float]:
    """Stratified by track: resample instances within each track, recompute the val_resolved delta."""
    rng = random.Random(seed)
    deltas = []
    tracks = sorted(pairs)
    for _ in range(iters):
        d = 0.0
        for t in tracks:
            v = pairs[t]
            n = len(v)
            idx = [rng.randrange(n) for _ in range(n)]
            d += (sum(v[i][1] - v[i][0] for i in idx) / n)
        deltas.append(d / len(tracks))
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="answer_match",
                    help="per-instance field to rank on; see COMPARABLE")
    ap.add_argument("--exp-glob", default="experiments/exp_Q*")
    ap.add_argument("--baseline", default="exp_Q000_baseline")
    ap.add_argument("--val", default="data/curated/curated_v3/validation.parquet")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if args.metric not in COMPARABLE:
        print(f"warning: {args.metric} is not in {COMPARABLE}; if it depends on the reward config "
              f"the arms are not comparable on it")
    binary = args.metric in ("answer_match",)
    vids = val_instance_ids(os.path.join(ROOT, args.val))
    arms: dict[str, tuple[int, dict[str, tuple[str, float]]]] = {}
    for d in sorted(glob.glob(os.path.join(ROOT, args.exp_glob))):
        if not os.path.isdir(d):
            continue
        step, sc = load_arm(d, vids, args.metric)
        if sc:
            arms[os.path.basename(d)] = (step, sc)

    if args.baseline not in arms:
        print(f"baseline {args.baseline} has no validation rows; nothing to compare against")
        return 1

    b_step, b_sc = arms[args.baseline]
    b_res, b_rates = resolved(b_sc)
    print(f"validation instances in split: {len(vids)}   metric: {args.metric}")
    print(f"baseline {args.baseline} (step {b_step}): {b_res:.4f}")
    for t, (r, n) in sorted(b_rates.items()):
        print(f"    {t:<9} {r:.4f}  n={n}" + (f"  binomial SE {binom_se(r, n):.4f}" if binary else ""))
    floor = math.sqrt(sum(binom_se(r, n) ** 2 for r, n in b_rates.values())) / len(b_rates)
    if binary:
        print(f"    -> SE of the mean: {floor:.4f}  (a gap below ~{2*floor:.3f} is not measurable unpaired)")
    print()

    out = {"metric": args.metric, "baseline": args.baseline, "baseline_value": b_res,
           "unpaired_se": floor if binary else None, "arms": {}}
    hdr = (f"{'arm':<24} {'step':>4} {'value':>9} {'delta':>8} {'95% CI (paired)':>20} "
           + (f"{'McNemar p':>10}  discordant" if binary else f"{'better':>8} {'worse':>7}"))
    print(hdr)
    print("-" * len(hdr))
    for name in sorted(arms):
        if name == args.baseline:
            continue
        step, sc = arms[name]
        shared = sorted(set(sc) & set(b_sc))
        if not shared:
            print(f"{name:<24} {step:>4}  no shared validation instances")
            continue
        a_res, a_rates = resolved({i: sc[i] for i in shared})
        pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
        b_wins = c_wins = 0
        for i in shared:
            t, a_ok = sc[i]
            _, b_ok = b_sc[i]
            pairs[t].append((b_ok, a_ok))
            if a_ok > b_ok:
                b_wins += 1
            elif b_ok > a_ok:
                c_wins += 1
        lo, hi = paired_bootstrap(pairs, args.iters, args.seed)
        p = mcnemar_exact(b_wins, c_wins) if binary else float("nan")
        delta = a_res - b_res
        tail = (f"{p:>10.3f}  {b_wins} win / {c_wins} lose of {len(shared)}"
                if binary else f"{b_wins:>8} {c_wins:>7}")
        print(f"{name:<24} {step:>4} {a_res:>9.4f} {delta:>+8.4f}  [{lo:>+.4f}, {hi:>+.4f}] {tail}")
        out["arms"][name] = {"step": step, "resolved": a_res, "delta": delta, "ci95": [lo, hi],
                             "mcnemar_p": p, "wins": b_wins, "losses": c_wins, "n_shared": len(shared),
                             "per_track": {t: {"rate": r, "n": n} for t, (r, n) in a_rates.items()}}

    print()
    sig = [n for n, v in out["arms"].items() if v["ci95"][0] > 0 or v["ci95"][1] < 0]
    print(f"arms whose 95% CI excludes 0: {sig if sig else 'none - every arm is tied with the baseline'}")

    if args.json_out:
        p = os.path.join(ROOT, args.json_out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
