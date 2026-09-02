#!/usr/bin/env python3
"""Build the curated RL dataset (spec sections 12, 16, 17).

Inputs (immutable): sources/rl_code_v1/data/parquet/*.parquet, sources/rl_code_v1/data/index/*.jsonl,
environments/manifests/validation_swe.jsonl, environments/manifests/validation_ut.jsonl,
optional --phat <jsonl> with empirical success rates from our own rollouts (instance_id, p_hat, n).

Outputs: data/curated/<version>/{train,validation,test,excluded}.parquet, dataset_manifest.json,
         data/processed/index/<version>/  (augmented index for RL_INDEX_DIR: f2p_effective / p2p_effective)
Raw rows are never modified; exclusions are recorded with reason codes.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, random, shutil, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{ROOT}/sources/rl_code_v1/data"
MAN = f"{ROOT}/environments/manifests"


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="curated_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phat", default="")
    ap.add_argument("--band", default="0.05,0.95", help="keep instances with measured p_hat in [lo,hi] (only when --phat given)")
    ap.add_argument("--n_val_ut", type=int, default=400)
    ap.add_argument("--n_test_ut", type=int, default=600)
    ap.add_argument("--n_val_repos", type=int, default=3)
    ap.add_argument("--n_test_repos", type=int, default=3)
    ap.add_argument("--include_ext", action="store_true", help="include the 16,464 extended unit-test rows")
    ap.add_argument("--verifier_version", default="vf_v001")
    ap.add_argument("--reward_version", default="rw_v001_baseline")
    ap.add_argument("--rubric_version", default="rb_v001_code_hint")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out_dir = f"{ROOT}/data/curated/{a.version}"
    idx_dir = f"{ROOT}/data/processed/index/{a.version}"
    os.makedirs(out_dir, exist_ok=True); os.makedirs(idx_dir, exist_ok=True)

    # ------------------------------------------------------------------ load
    frames = [pd.read_parquet(f"{SRC}/parquet/t15_code_rl_core_train.parquet"), pd.read_parquet(f"{SRC}/parquet/t15_code_rl_core_val.parquet")]
    if a.include_ext:
        frames.append(pd.read_parquet(f"{SRC}/parquet/t15_code_rl_extended_train.parquet"))
    df = pd.concat(frames, ignore_index=True)
    ei = pd.DataFrame([dict(x) for x in df["extra_info"]])
    df["_iid"] = ei["instance_id"].values; df["_variant"] = ei["variant_type"].values; df["_repo"] = ei["repo"].values
    df["_lang"] = ei["lang"].values; df["_harness"] = ei["harness"].values; df["_p_hat_src"] = ei["p_hat_source"].values
    df["_prompt_text"] = [json.dumps([dict(m) for m in p], ensure_ascii=False) for p in df["prompt"]]
    vswe = {r["instance_id"]: r for r in load_jsonl(f"{MAN}/validation_swe.jsonl")}
    vut = {r["instance_id"]: r for r in load_jsonl(f"{MAN}/validation_ut.jsonl")}
    phat = {}
    if a.phat:
        for r in load_jsonl(a.phat):
            phat[r["instance_id"]] = r
    lo, hi = (float(x) for x in a.band.split(","))
    swe_full = {r["instance_id"]: r for r in load_jsonl(f"{SRC}/index/t15_code_swesmith.full.jsonl")}

    # ------------------------------------------------------------------ per-row quality decision
    seen_prompt = {}
    status, reason, flags_col, meta_col = [], [], [], []
    for i, row in df.iterrows():
        iid, var = row["_iid"], row["_variant"]
        flags, excl = [], ""
        meta = {"environment_id": "", "f2p_effective": [], "p2p_effective": [], "n_f2p_effective": 0, "n_p2p_effective": 0,
                "empirical_success_rate": None, "empirical_n": 0}
        key = sha16(row["_prompt_text"])
        if key in seen_prompt:
            excl = excl or "DUPLICATE"; flags.append(f"dup_of:{seen_prompt[key]}")
        else:
            seen_prompt[key] = iid
        if var.startswith("swe"):
            meta["environment_id"] = swe_full.get(iid, {}).get("image_name", "").rsplit("/", 1)[-1]
            v = vswe.get(iid)
            if var != "swe_32k":
                excl = excl or "CONTEXT_OVERFLOW"; flags.append("prompt_gt_32k")
            if v is None:
                excl = excl or "UNVALIDATED"
            else:
                g, e = v.get("gold", {}), v.get("empty", {})
                if not v.get("extract_ok"):
                    excl = excl or "BROKEN_ENV"
                elif e.get("ran") and e.get("n_f2p", 0) > 0:
                    excl = excl or ("BUG_IN_TEST_FILE" if v.get("gold_touches_tests") else "BASELINE_ANOMALY")
                elif g.get("apply") == "patch_apply_fail":
                    tail = (g.get("log_tail") or "")
                    excl = excl or ("BUG_IN_TEST_FILE" if v.get("gold_touches_tests") else
                                    ("BROKEN_BRANCH" if "No such file" in tail else "REFERENCE_PATCH_INVALID"))
                elif not g.get("ran"):
                    excl = excl or "BROKEN_ENV"; flags.append(g.get("err_kind", ""))
                elif not v.get("f2p_effective"):
                    excl = excl or "NO_VALID_F2P"
                if v.get("gold_needs_ignore_ws"):
                    flags.append("GOLD_WHITESPACE_MISMATCH")
                if v.get("gold_touches_tests"):
                    flags.append("gold_touches_tests")
                meta["f2p_effective"] = v.get("f2p_effective") or []
                meta["p2p_effective"] = v.get("p2p_effective") or []
                meta["n_f2p_effective"] = len(meta["f2p_effective"]); meta["n_p2p_effective"] = len(meta["p2p_effective"])
                if v.get("f2p_dropped"):
                    flags.append(f"F2P_PRUNED:{len(v['f2p_dropped'])}")
                if v.get("p2p_dropped"):
                    flags.append(f"P2P_PRUNED:{len(v['p2p_dropped'])}")
            jv = ei.loc[i].get("judge_verdict", None)
            if jv == "drop":
                flags.append("JUDGE_DROP")
        else:
            meta["environment_id"] = "python_3.11-slim-bookworm+ut_venv"
            v = vut.get(iid)
            if v is None:
                excl = excl or "UNVALIDATED"
            else:
                if v.get("n_tests", 0) == 0:
                    excl = excl or "NO_TESTS"
                elif v.get("has_reference") and not v.get("gold_ok", False):
                    excl = excl or "REFERENCE_SOLUTION_FAILS"
                elif not v.get("empty_ok", True):
                    e = v.get("empty", {})
                    frac_empty = (e.get("n_f2p", 0) / e.get("t_f2p", 1)) if e.get("t_f2p") else 0.0
                    if frac_empty >= 0.5:
                        excl = excl or "BASELINE_ANOMALY"      # a no-op solution already passes half the tests
                    else:
                        flags.append(f"EMPTY_PASSES_SOME:{e.get('n_f2p', 0)}/{e.get('t_f2p', 0)}")
                if not v.get("has_reference"):
                    flags.append("no_reference_solution")
        if phat:
            p = phat.get(iid)
            if p is not None:
                meta["empirical_success_rate"] = float(p["p_hat"]); meta["empirical_n"] = int(p.get("n", 0))
                if p["p_hat"] < lo:
                    excl = excl or "TOO_HARD"
                elif p["p_hat"] > hi:
                    excl = excl or "TOO_EASY"
        status.append("excluded" if excl else "ok"); reason.append(excl); flags_col.append(flags); meta_col.append(meta)
    df["_status"], df["_reason"], df["_flags"], df["_meta"] = status, reason, flags_col, meta_col

    # ------------------------------------------------------------------ splits
    ok = df[df["_status"] == "ok"].copy()
    swe = ok[ok["_variant"].str.startswith("swe")]
    ut = ok[~ok["_variant"].str.startswith("swe")]
    repos = sorted(swe["_repo"].unique()); rng.shuffle(repos)
    # balance: pick val/test repos alternately from a shuffled list, preferring medium-sized repos
    counts = swe["_repo"].value_counts().to_dict()
    ordered = sorted(repos, key=lambda r: (-counts[r], r))
    rng.shuffle(ordered)
    test_repos, val_repos = ordered[: a.n_test_repos], ordered[a.n_test_repos: a.n_test_repos + a.n_val_repos]
    split = {}
    for iid, repo in zip(swe["_iid"], swe["_repo"]):
        split[iid] = "test" if repo in test_repos else ("validation" if repo in val_repos else "train")
    ut_ids = list(ut["_iid"]); rng.shuffle(ut_ids)
    for j, iid in enumerate(ut_ids):
        split[iid] = "test" if j < a.n_test_ut else ("validation" if j < a.n_test_ut + a.n_val_ut else "train")
    df["_split"] = [split.get(i, "excluded") for i in df["_iid"]]

    # ------------------------------------------------------------------ write
    def enrich(row):
        e = dict(row["extra_info"])
        m = row["_meta"]
        e.update({"environment_id": m["environment_id"], "f2p_effective": list(m["f2p_effective"]), "p2p_effective": list(m["p2p_effective"]),
                  "n_f2p_effective": int(m["n_f2p_effective"]), "n_p2p_effective": int(m["n_p2p_effective"]),
                  "empirical_success_rate": -1.0 if m["empirical_success_rate"] is None else float(m["empirical_success_rate"]),
                  "empirical_n": int(m["empirical_n"]), "data_quality_status": row["_status"], "exclusion_reason": row["_reason"],
                  "quality_flags": ";".join(row["_flags"]), "verifier_version": a.verifier_version, "reward_version": a.reward_version,
                  "rubric_version": a.rubric_version, "dataset_version": a.version, "split": row["_split"],
                  "task_type": "swe_patch" if row["_variant"].startswith("swe") else f"unittest_{row['_harness']}",
                  "difficulty": e.get("difficulty_band", ""), "base_commit": (row["_iid"].split(".")[1] if row["_variant"].startswith("swe") and "." in row["_iid"] else ""),
                  "branch": row["_iid"] if row["_variant"].startswith("swe") else ""})
        return e
    df["extra_info"] = [enrich(r) for _, r in df.iterrows()]
    cols = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    manifest = {"version": a.version, "seed": a.seed, "include_ext": a.include_ext, "phat_file": a.phat, "band": [lo, hi],
                "counts": {}, "exclusions": {}, "flags": {}, "swe_repos": {"train": sorted(set(repos) - set(test_repos) - set(val_repos)),
                                                                           "validation": val_repos, "test": test_repos},
                "verifier_version": a.verifier_version, "reward_version": a.reward_version, "rubric_version": a.rubric_version}
    for name in ("train", "validation", "test"):
        part = df[df["_split"] == name]
        part[cols].to_parquet(f"{out_dir}/{name}.parquet", index=False)
        manifest["counts"][name] = {"total": int(len(part)), **{k: int(v) for k, v in part["_variant"].value_counts().items()}}
    exc = df[df["_status"] == "excluded"]
    exc_out = exc[cols].copy(); exc_out["exclusion_reason"] = exc["_reason"].values; exc_out["instance_id"] = exc["_iid"].values
    exc_out.to_parquet(f"{out_dir}/excluded.parquet", index=False)
    manifest["counts"]["excluded"] = int(len(exc))
    manifest["exclusions"] = {k: int(v) for k, v in collections.Counter(exc["_reason"]).items()}
    manifest["exclusions_by_variant"] = {k: int(v) for k, v in collections.Counter(zip(exc["_variant"], exc["_reason"])).items()} if len(exc) else {}
    manifest["flags"] = {k: int(v) for k, v in collections.Counter(f.split(":")[0] for fl in df["_flags"] for f in fl).items()}
    for name in ("train", "validation", "test", "excluded"):
        with open(f"{out_dir}/{name}.parquet", "rb") as f:
            manifest.setdefault("sha256", {})[name] = hashlib.sha256(f.read()).hexdigest()[:16]
    # augmented index for the runner (RL_INDEX_DIR)
    aug = []
    for iid, r in swe_full.items():
        v = vswe.get(iid, {})
        r2 = dict(r); r2["f2p_effective"] = v.get("f2p_effective") or []; r2["p2p_effective"] = v.get("p2p_effective") or []
        aug.append(r2)
    with open(f"{idx_dir}/t15_code_swesmith.full.jsonl", "w", encoding="utf-8") as f:
        for r in aug:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for fn in os.listdir(f"{SRC}/index"):
        if fn.endswith(".jsonl") and fn != "t15_code_swesmith.full.jsonl":
            shutil.copy(f"{SRC}/index/{fn}", f"{idx_dir}/{fn}")
    manifest["index_dir"] = idx_dir
    json.dump(manifest, open(f"{out_dir}/dataset_manifest.json", "w"), indent=1, ensure_ascii=False)
    # per-instance audit table
    audit = pd.DataFrame({"instance_id": df["_iid"], "variant_type": df["_variant"], "repo": df["_repo"], "lang": df["_lang"],
                          "split": df["_split"], "status": df["_status"], "exclusion_reason": df["_reason"],
                          "flags": [";".join(f) for f in df["_flags"]], "n_f2p_effective": [m["n_f2p_effective"] for m in df["_meta"]],
                          "n_p2p_effective": [m["n_p2p_effective"] for m in df["_meta"]], "p_hat_orig": ei["p_hat"].values,
                          "p_hat_source_orig": df["_p_hat_src"], "empirical_success_rate": [m["empirical_success_rate"] for m in df["_meta"]]})
    audit.to_csv(f"{out_dir}/instance_audit.csv", index=False)
    print(json.dumps({k: manifest[k] for k in ("counts", "exclusions", "flags", "swe_repos")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
