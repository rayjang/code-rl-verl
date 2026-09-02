#!/usr/bin/env python3
"""Experiment registry (autoresearch loop backbone).

experiments/registry.jsonl   one JSON record per experiment (append-only; latest record per id wins)
experiments/<exp_id>/        env.sh, overrides.txt, train.log, instance_log.jsonl, metrics.json, summary.json

Commands
  python scripts/experiment.py new --id EXP --parent PARENT --hypothesis "..." --changes k=v,... \
        --reward configs/reward/x.yaml --verifier configs/verifier/y.yaml --rubric rubrics/z.yaml \
        --train data/curated/train.parquet --val data/curated/validation.parquet --overrides-from experiments/base/overrides.txt \
        --set trainer.total_training_steps=40 --set ... --seed 0
  python scripts/experiment.py submit --id EXP [--cpus 40 --mem 500G --time 06:00:00]
  python scripts/experiment.py collect --id EXP            # parse train.log -> metrics.json + summary.json
  python scripts/experiment.py decide --id EXP --keep|--reject --reason "..."
  python scripts/experiment.py list
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(ROOT, "experiments", "registry.jsonl")
S_MODEL = "/scratch/r919a03/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be"


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def sha(path):
    if not path or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_registry():
    recs = {}
    if os.path.exists(REG):
        for line in open(REG, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                recs[d["id"]] = {**recs.get(d["id"], {}), **d}
    return recs


def append(rec):
    os.makedirs(os.path.dirname(REG), exist_ok=True)
    with open(REG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def yaml_version(path):
    if not path or not os.path.exists(path):
        return ""
    for line in open(path, encoding="utf-8"):
        if line.startswith("version:"):
            return line.split(":", 1)[1].strip().strip('"')
    return os.path.basename(path)


# ----------------------------------------------------------------------------------------------- new
def cmd_new(a):
    exp_dir = os.path.join(ROOT, "experiments", a.id)
    os.makedirs(exp_dir, exist_ok=True)
    base = [l.rstrip("\n") for l in open(a.overrides_from, encoding="utf-8")] if a.overrides_from else []
    sets = {}
    for s in a.set or []:
        k, v = s.split("=", 1)
        sets[k] = v
    if a.train:
        sets["data.train_files"] = a.train
    if a.val:
        sets["data.val_files"] = a.val
    sets.setdefault("data.seed", str(a.seed))
    sets["trainer.experiment_name"] = "$EXP_NAME"
    sets["trainer.default_local_dir"] = "$ROOT/checkpoints/$EXP_NAME"
    out, seen = [], set()
    for line in base:
        if not line.strip() or line.startswith("#"):
            out.append(line); continue
        k = line.lstrip("+").split("=", 1)[0]
        if k in sets:
            out.append(("+" if line.startswith("+") else "") + f"{k}={sets[k]}"); seen.add(k)
        else:
            out.append(line)
    for k, v in sets.items():
        if k not in seen:
            out.append(f"{k}={v}")
    with open(os.path.join(exp_dir, "overrides.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(exp_dir, "env.sh"), "w", encoding="utf-8") as f:
        f.write(f"export EXP_NAME={a.id}\nexport MODEL_PATH={S_MODEL}\n")
        f.write(f"export REWARD_CONFIG=$ROOT/{a.reward}\nexport VERIFIER_CONFIG=$ROOT/{a.verifier}\n")
        if a.rubric:
            f.write(f"export RUBRIC_CONFIG=$ROOT/{a.rubric}\n")
        f.write(f"export VERIFIER_CONCURRENCY={a.concurrency}\n")
    rec = {"id": a.id, "parent": a.parent, "hypothesis": a.hypothesis, "changed_variables": a.changes,
           "created_at": now(), "git_commit": git_sha(), "seed": a.seed,
           "dataset_version": {"train": a.train, "train_sha": sha(a.train), "val": a.val, "val_sha": sha(a.val)},
           "reward_version": yaml_version(a.reward), "verifier_version": yaml_version(a.verifier),
           "rubric_version": yaml_version(a.rubric) if a.rubric else "rb_v000_none",
           "reward_config": a.reward, "verifier_config": a.verifier, "rubric_config": a.rubric,
           "overrides": out, "status": "created", "gpu": "gpu48:6xH200", "stage": a.stage}
    append(rec)
    print(json.dumps({k: rec[k] for k in ("id", "parent", "git_commit", "reward_version", "verifier_version", "rubric_version")}))


# -------------------------------------------------------------------------------------------- submit
def cmd_submit(a):
    exp_dir = os.path.join(ROOT, "experiments", a.id)
    cmd = ["sbatch", "-c", str(a.cpus), f"--mem={a.mem}", "-t", a.time, "-J", a.id[:12], os.path.join(ROOT, "scripts/launch_verl.sh"), exp_dir]
    out = subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
    job = re.findall(r"\d+", out)[-1]
    append({"id": a.id, "status": "submitted", "slurm_job": job, "submitted_at": now(), "git_commit": git_sha()})
    print(out)


# ------------------------------------------------------------------------------------------- collect
STEP_RE = re.compile(r"step:(\d+) - (.*)$")
KV_RE = re.compile(r"([A-Za-z0-9_/.@\-]+):(-?[0-9.]+(?:[eE][-+]?\d+)?|nan|inf|-inf)")


def parse_train_log(path):
    steps = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        d = steps.setdefault(step, {})
        for k, v in KV_RE.findall(m.group(2)):
            try:
                d[k] = float(v)
            except ValueError:
                pass
    return steps


KEYS = {"reward": "critic/score/mean", "reward_std": "critic/score/std", "pg_loss": "actor/pg_loss", "kl": "actor/ppo_kl",
        "entropy": "actor/entropy", "grad_norm": "actor/grad_norm", "lr": "actor/lr", "resp_len": "response_length/mean",
        "clipfrac": "actor/pg_clipfrac"}


def cmd_collect(a):
    exp_dir = os.path.join(ROOT, "experiments", a.id)
    log = os.path.join(exp_dir, "train.log")
    steps = parse_train_log(log) if os.path.exists(log) else {}
    inst_log = os.path.join(exp_dir, "instance_log.jsonl")
    inst_summary = {}
    if os.path.exists(inst_log):
        import collections
        agg = collections.defaultdict(lambda: collections.defaultdict(float)); n = collections.Counter()
        for line in open(inst_log, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = (r.get("step", 0), r.get("track", ""))
            n[k] += 1
            for f in ("final_reward", "rule_correctness_score", "f2p_frac", "p2p_frac", "patch_format_score", "patch_apply_score",
                      "infra_excluded", "rubric_score", "gated_out"):
                agg[k][f] += float(r.get(f, 0) or 0)
        inst_summary = {f"{s}:{t}": {"n": n[(s, t)], **{f: v / n[(s, t)] for f, v in agg[(s, t)].items()}} for (s, t) in sorted(n)}
    info = {}
    ri = os.path.join(exp_dir, "run_info.txt")
    if os.path.exists(ri):
        txt = open(ri).read()
        m = re.search(r"END_TRAIN rc=(\d+) (\S+)", txt)
        info = {"end_rc": int(m.group(1)) if m else None, "end_time": m.group(2) if m else None,
                "start_time": (re.search(r"start=(\S+)", txt) or [None, None])[1], "git": (re.search(r"git=(\S+)", txt) or [None, None])[1]}
    val = {k: v for s in steps.values() for k, v in s.items() if k.startswith("val-core") or k.startswith("val/")}
    curves = {str(s): {name: steps[s].get(key) for name, key in KEYS.items()} for s in sorted(steps)}
    metrics = {"steps": steps, "curves": curves, "instance_summary": inst_summary, "run_info": info, "val": val}
    json.dump(metrics, open(os.path.join(exp_dir, "metrics.json"), "w"), indent=1)
    last = steps[max(steps)] if steps else {}
    summary = {"id": a.id, "n_steps": len(steps), "final_reward": last.get(KEYS["reward"]), "final_kl": last.get(KEYS["kl"]),
               "final_entropy": last.get(KEYS["entropy"]), "max_grad_norm": max((s.get(KEYS["grad_norm"], 0) for s in steps.values()), default=None),
               "val": val, "end_rc": info.get("end_rc"), "collected_at": now()}
    append({"id": a.id, "status": "collected" if info.get("end_rc") == 0 else "failed_or_running", "summary": summary,
            "start_time": info.get("start_time"), "end_time": info.get("end_time")})
    print(json.dumps(summary, indent=1)[:3000])


def cmd_decide(a):
    append({"id": a.id, "decision": "keep" if a.keep else "reject", "decision_reason": a.reason, "decided_at": now()})
    print("recorded")


def cmd_list(a):
    for i, r in load_registry().items():
        s = r.get("summary", {})
        print(f"{i:40s} parent={r.get('parent','')!s:22s} status={r.get('status','')!s:18s} decision={r.get('decision','')!s:7s} "
              f"reward={s.get('final_reward')} val={ {k: round(v,4) for k,v in (s.get('val') or {}).items() if 'score' in k or 'resolved' in k} }")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new"); p.add_argument("--id", required=True); p.add_argument("--parent", default=""); p.add_argument("--hypothesis", default="")
    p.add_argument("--changes", default=""); p.add_argument("--reward", default="configs/reward/rw_v001_baseline.yaml")
    p.add_argument("--verifier", default="configs/verifier/vf_v001.yaml"); p.add_argument("--rubric", default="")
    p.add_argument("--train", default=""); p.add_argument("--val", default=""); p.add_argument("--overrides-from", default="")
    p.add_argument("--set", action="append"); p.add_argument("--seed", type=int, default=0); p.add_argument("--stage", default="D")
    p.add_argument("--concurrency", type=int, default=6); p.set_defaults(fn=cmd_new)
    p = sub.add_parser("submit"); p.add_argument("--id", required=True); p.add_argument("--cpus", type=int, default=40)
    p.add_argument("--mem", default="500G"); p.add_argument("--time", default="06:00:00"); p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("collect"); p.add_argument("--id", required=True); p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("decide"); p.add_argument("--id", required=True); g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--keep", action="store_true"); g.add_argument("--reject", action="store_true"); p.add_argument("--reason", default="")
    p.set_defaults(fn=cmd_decide)
    p = sub.add_parser("list"); p.set_defaults(fn=cmd_list)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
