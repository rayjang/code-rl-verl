#!/usr/bin/env python3
"""Autoresearch loop (spec §13): baseline -> one hypothesis at a time -> controlled run -> compare -> keep/reject.

Plan file (YAML) example: configs/plans/stageD.yaml
  baseline: exp_D000_baseline
  primary_metric: val_score            # from experiment summary["val"] keys containing this substring (higher is better)
  guard_metrics: {max_grad_norm: 50, final_kl: 0.5}
  common: {overrides_from: experiments/base_stageD/overrides.txt, train: ..., val: ..., steps: 40, stage: D}
  hypotheses:
    - id: exp_D001_f2p_binary
      hypothesis: "all-or-nothing F2P gives sparser signal and worse held-out solve rate"
      reward: configs/reward/rw_v002_f2p_binary.yaml
      changes: "reward.f2p_mode=binary"
The loop is restartable: state = experiments/registry.jsonl (+ plan_state.json). Each iteration:
  create (if not created) -> submit (if not submitted) -> wait (poll sacct + run_info END_TRAIN) -> collect ->
  decide against the current best -> next hypothesis. `--dry` prints the plan.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from scripts.experiment import load_registry, append, now  # noqa: E402

PY = sys.executable


def sh(args):
    return subprocess.run([PY, f"{ROOT}/scripts/experiment.py"] + args, cwd=ROOT, capture_output=True, text=True)


def job_state(job):
    try:
        out = subprocess.check_output(["sacct", "-j", str(job), "--format=State", "-P", "-n"], text=True)
        states = [l.strip() for l in out.splitlines() if l.strip()]
        return states[0] if states else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def metric_of(summary, key):
    val = (summary or {}).get("val", {}) or {}
    cands = {k: v for k, v in val.items() if key in k}
    if not cands:
        return None
    # prefer exact 'val-core/.../score/mean'-like keys
    k = sorted(cands, key=len)[0]
    return float(cands[k])


def _failed(r, exp_id):
    ri = f"{ROOT}/experiments/{exp_id}/run_info.txt"
    if os.path.exists(ri):
        m = [l for l in open(ri) if "END_TRAIN" in l]
        if m and "rc=0" not in m[-1]:
            return True
        if m and "rc=0" in m[-1]:
            return False
    job = r.get("slurm_job")
    if job:
        st = job_state(job)
        if st.startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")):
            return True                      # job died without writing END_TRAIN (e.g. scancel)
    return r.get("status") in ("failed_or_running",) and (r.get("summary") or {}).get("end_rc") not in (0, None)


def ensure(exp, plan, common):
    reg = load_registry()
    r = reg.get(exp["id"], {})
    if r and _failed(r, exp["id"]) and r.get("decision") != "keep_final":
        # previous attempt failed: keep its record, re-create overrides (config may have been fixed) and resubmit
        attempt = int(r.get("attempt", 1)) + 1
        append({"id": exp["id"], "status": "retry", "attempt": attempt, "retried_at": now()})
        print(f"[{exp['id']}] previous attempt failed -> attempt {attempt}")
        r = {}
        for fn in ("run_info.txt", "train.log", "instance_log.jsonl", "metrics.json"):
            p = f"{ROOT}/experiments/{exp['id']}/{fn}"
            if os.path.exists(p):
                os.replace(p, p + f".attempt{attempt - 1}")
    if not r:
        args = ["new", "--id", exp["id"], "--parent", exp.get("parent", plan["baseline"]), "--hypothesis", exp.get("hypothesis", ""),
                "--changes", exp.get("changes", ""), "--reward", exp.get("reward", common.get("reward", "configs/reward/rw_v001_baseline.yaml")),
                "--verifier", exp.get("verifier", common.get("verifier", "configs/verifier/vf_v002.yaml")),
                "--train", exp.get("train", common["train"]), "--val", exp.get("val", common["val"]),
                "--overrides-from", exp.get("overrides_from", common["overrides_from"]), "--seed", str(exp.get("seed", common.get("seed", 0))),
                "--stage", common.get("stage", "D")]
        if exp.get("rubric") or common.get("rubric"):
            args += ["--rubric", exp.get("rubric", common.get("rubric"))]
        if exp.get("model") or common.get("model"):
            args += ["--model", exp.get("model", common.get("model"))]
        for s in (common.get("set", []) + exp.get("set", [])):
            args += ["--set", s]
        p = sh(args); print(p.stdout.strip(), p.stderr.strip()[-300:])
        reg = load_registry(); r = reg[exp["id"]]
    if r.get("status") == "created" or ("slurm_job" not in r and r.get("status") not in ("collected",)):
        p = sh(["submit", "--id", exp["id"], "--cpus", str(common.get("cpus", 40)), "--mem", common.get("mem", "480G"), "--time", common.get("time", "08:00:00"),
                "--gpus", str(common.get("gpus", int(os.environ.get("NGPU", "4"))))])
        print(p.stdout.strip(), p.stderr.strip()[-300:])
    return load_registry()[exp["id"]]


def wait(exp_id, poll=120):
    while True:
        r = load_registry()[exp_id]
        ri = f"{ROOT}/experiments/{exp_id}/run_info.txt"
        if os.path.exists(ri) and "END_TRAIN" in open(ri).read():
            return
        st = job_state(r.get("slurm_job", ""))
        if st.startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")):
            print(f"[{exp_id}] slurm state {st}")
            return
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--plan", required=True); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated hypothesis ids to run")
    a = ap.parse_args()
    plan = yaml.safe_load(open(a.plan)); common = plan.get("common", {})
    state_path = f"{ROOT}/experiments/plan_state_{os.path.basename(a.plan).split('.')[0]}.json"
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"best": plan["baseline"], "done": []}
    todo = [{"id": plan["baseline"], **plan.get("baseline_spec", {})}] + plan["hypotheses"]
    if a.only:
        keep = set(a.only.split(",")); todo = [t for t in todo if t["id"] in keep or t["id"] == plan["baseline"]]
    if a.dry:
        print(json.dumps(todo, indent=1, ensure_ascii=False)); return
    for exp in todo:
        if exp["id"] in state["done"]:
            continue
        print(f"=== {exp['id']} (best so far: {state['best']}) ===", flush=True)
        ensure(exp, plan, common)
        wait(exp["id"])
        p = sh(["collect", "--id", exp["id"]]); print(p.stdout[-1500:])
        reg = load_registry(); me = reg[exp["id"]].get("summary", {}) or {}
        pm = plan.get("primary_metric", "score")
        mine = metric_of(me, pm)
        if exp["id"] == plan["baseline"]:
            if mine is None or reg[exp["id"]].get("summary", {}).get("end_rc") not in (0,):
                sh(["decide", "--id", exp["id"], "--reject", "--reason", "baseline run failed; plan aborted"])
                print("BASELINE FAILED -> aborting plan", flush=True)
                return
            decision, reason = "keep", "baseline"
        else:
            best = metric_of(reg[state["best"]].get("summary", {}), pm)
            guards = plan.get("guard_metrics", {}) or {}
            guard_fail = [k for k, lim in guards.items() if me.get(k) is not None and me[k] > lim]
            if mine is None or reg[exp["id"]].get("summary", {}).get("end_rc") not in (0, None):
                decision, reason = "reject", "run failed or no validation metric"
            elif guard_fail:
                decision, reason = "reject", f"guard metrics violated: {guard_fail}"
            elif best is None or mine > best + float(plan.get("min_improvement", 0.0)):
                decision, reason = "keep", f"{pm} {mine:.4f} > best {best}"
            else:
                decision, reason = "reject", f"{pm} {mine} <= best {best}"
        sh(["decide", "--id", exp["id"], f"--{decision}", "--reason", reason])
        print(f"[{exp['id']}] {decision}: {reason}", flush=True)
        if decision == "keep" and exp["id"] != plan["baseline"] and not plan.get("independent_arms", True):
            state["best"] = exp["id"]
        elif decision == "keep" and exp["id"] != plan["baseline"] and plan.get("independent_arms", True):
            # ablation mode: every arm is compared with the baseline; best tracks the highest metric
            if mine is not None and (metric_of(reg[state["best"]].get("summary", {}), pm) or -1) < mine:
                state["best"] = exp["id"]
        state["done"].append(exp["id"]); json.dump(state, open(state_path, "w"), indent=1)
    print("PLAN DONE; best =", state["best"])


if __name__ == "__main__":
    main()
