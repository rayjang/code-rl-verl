#!/usr/bin/env python3
"""Environment validation (spec section 19) for every SWE-smith instance and a unit-test sample.

For each SWE instance:  base-tree extraction (clean checkout + image launch) -> empty patch run
(baseline: F2P must FAIL, P2P must PASS) -> gold patch run (F2P must PASS, P2P must PASS) -> cleanup.
Records one JSON line per instance in environments/manifests/validation_swe.jsonl.
For unit-test instances: reference solution must pass all tests; empty code must not.

Usage: python scripts/validate_environments.py --track swe [--limit N] [--workers 16]
       python scripts/validate_environments.py --track ut --limit 400
"""
import argparse, json, os, sys, time, random, collections
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from verifier.swe_runner import SweSmithRunner
from verifier.ut_runner import UnitTestRunner
from verifier.diff_parser import touched_paths
from verifier.anti_hacking import classify_paths  # noqa

IDX = os.path.join(ROOT, "sources/rl_code_v1/data/index")
SIF_DIR = os.path.join(ROOT, "environments/sif")
OUT_DIR = os.path.join(ROOT, "environments/manifests")


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def summarize(er):
    return {"ran": er.ran, "err_kind": er.err_kind.value, "err": er.err[:200], "n_f2p": er.f2p.n_passed, "t_f2p": er.f2p.total,
            "f2p_failed": list(er.f2p.failed)[:50], "p2p_failed": list(er.p2p.failed)[:50], "p2p_missing_ids": list(er.p2p.missing)[:50],
            "f2p_missing": len(er.f2p.missing), "n_p2p": er.p2p.n_passed, "t_p2p": er.p2p.total, "p2p_missing": len(er.p2p.missing),
            "resolved": er.resolved, "runtime_s": round(er.runtime_s, 1), "timed_out": er.timed_out,
            "apply": er.apply.status.value, "log_tail": er.log_tail[-600:] if not er.ran else ""}


def validate_swe(inst, runner):
    t0 = time.time()
    rec = {"instance_id": inst["instance_id"], "repo": inst["repo"], "image_name": inst["image_name"], "sif": runner.sif_for(inst),
           "n_f2p_index": len(inst.get("FAIL_TO_PASS") or []), "n_p2p_index": len(inst.get("PASS_TO_PASS") or [])}
    tar, err = runner.base_tar(inst)
    rec["extract_ok"] = tar is not None
    rec["extract_err"] = err
    if tar is None:
        rec["elapsed_s"] = round(time.time() - t0, 1)
        return rec
    gold = inst.get("gold_patch") or ""
    rec["gold_touches"] = list(touched_paths(gold))
    rec["gold_touches_tests"] = bool(classify_paths(touched_paths(gold), inst)["test_paths"])
    rec["empty"] = summarize(runner.run(inst, ""))
    rec["gold"] = summarize(runner.run(inst, gold))
    e, g = rec["empty"], rec["gold"]
    rec["baseline_ok"] = bool(e["ran"] and e["n_f2p"] == 0 and e["n_p2p"] == e["t_p2p"])
    rec["gold_ok"] = bool(g["ran"] and g["resolved"])
    rec["status"] = "ok" if (rec["baseline_ok"] and rec["gold_ok"]) else (
        "gold_fail" if not rec["gold_ok"] else "baseline_anomaly")
    rec["elapsed_s"] = round(time.time() - t0, 1)
    return rec


def validate_ut(inst, runner):
    t0 = time.time()
    ref = inst.get("reference_solution") or ""
    rec = {"instance_id": inst["instance_id"], "harness": inst.get("harness"), "n_tests": len(inst.get("tests") or []),
           "has_reference": bool(ref.strip())}
    if ref.strip():
        rec["gold"] = summarize(runner.run(inst, ref))
        rec["gold_ok"] = rec["gold"]["resolved"]
    rec["empty"] = summarize(runner.run(inst, "pass\n"))
    rec["empty_ok"] = rec["empty"]["ran"] and rec["empty"]["n_f2p"] == 0
    rec["status"] = "ok" if (rec.get("gold_ok", True) and rec["empty_ok"]) else ("gold_fail" if not rec.get("gold_ok", True) else "baseline_anomaly")
    rec["elapsed_s"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["swe", "ut"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default=os.environ.get("VERIFIER_RUN_DIR", "/tmp/r919a03_verifier_run"))
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "environments/cache_v2"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = a.out or os.path.join(OUT_DIR, f"validation_{a.track}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["instance_id"] for l in open(out_path)}
    if a.track == "swe":
        insts = load_jsonl(f"{IDX}/t15_code_swesmith.full.jsonl")
        runner = SweSmithRunner(SIF_DIR, a.run_dir, a.cache_dir, timeout=900)
        fn = lambda i: validate_swe(i, runner)
    else:
        core = load_jsonl(f"{IDX}/t15_code_unittest.full.jsonl")
        ext = load_jsonl(f"{IDX}/t15_code_unittest_ext.full.jsonl")
        random.Random(a.seed).shuffle(core); random.Random(a.seed + 1).shuffle(ext)
        insts = core + ext
        runner = UnitTestRunner(os.path.join(SIF_DIR, "python_3.11-slim-bookworm.sif"), os.path.join(ROOT, "environments/ut_venv"), a.run_dir)
        fn = lambda i: validate_ut(i, runner)
    insts = [i for i in insts if i["instance_id"] not in done]
    if a.limit:
        insts = insts[: a.limit]
    print(f"{a.track}: {len(insts)} instances to validate ({len(done)} already done) -> {out_path}", flush=True)
    stats = collections.Counter()
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex, open(out_path, "a", encoding="utf-8") as f:
        futs = {ex.submit(fn, i): i["instance_id"] for i in insts}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"instance_id": futs[fut], "status": "exception", "error": repr(e)[:300]}
            stats[rec.get("status", "?")] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if n % 20 == 0 or n == len(insts):
                print(f"[{n}/{len(insts)}] {dict(stats)} elapsed={time.time() - t0:.0f}s", flush=True)
    print("DONE", dict(stats))


if __name__ == "__main__":
    main()
