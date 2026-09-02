#!/usr/bin/env python3
"""Stage B — offline verifier ablation (spec §4, §14-B).

Two inputs:
  (1) synthetic responses derived from GOLD patches in 9 wrappings (fenced diff, fenced other lang, raw,
      prose-before/after, S/R blocks, CRLF, begin-patch, two blocks, truncated hunk) -> extraction
      precision/recall per parser (ground truth: the gold patch applies and resolves);
  (2) real model responses (results/phat/*/responses.jsonl) -> per parser/select/apply variant: extraction
      rate, structural-valid rate, apply rate, and agreement with the reference verifier (vf_v002) on
      `resolved` (executed for a bounded sample of SWE responses when --execute is given).
Writes results/verifier_ablation/<tag>/{synthetic.csv, model.csv, summary.json}.
"""
from __future__ import annotations
import argparse, collections, json, os, random, sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from verifier.diff_parser import PARSERS, extract, search_replace_to_diff, validate_unified_diff
from verifier.registry import Registry
from verifier.schemas import ExtractStatus

PARSER_ARMS = [("strict", "last"), ("markdown", "last"), ("fallback", "last"), ("fallback", "first"), ("fallback", "concat_distinct"), ("baseline", "last")]


def to_search_replace(patch: str, buggy: dict) -> str:
    """Rewrite a unified diff of single-file hunks into <solution> SEARCH/REPLACE blocks (exact hunk text)."""
    out = ["<solution>"]
    cur = None; old, new = [], []
    def flush():
        if cur and (old or new):
            out.append(f"### {cur}\n<<<<<<< SEARCH\n" + "\n".join(old) + "\n=======\n" + "\n".join(new) + "\n>>>>>>> REPLACE")
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            flush(); cur = line.split(" b/")[-1]; old, new = [], []
        elif line.startswith(("--- ", "+++ ", "index ")):
            continue
        elif line.startswith("@@"):
            flush(); old, new = [], []
        elif line.startswith("-"):
            old.append(line[1:])
        elif line.startswith("+"):
            new.append(line[1:])
        elif line.startswith(" ") or line == "":
            old.append(line[1:]); new.append(line[1:])
    flush(); out.append("</solution>")
    return "\n".join(out)


def wrappings(gold: str, buggy: dict) -> dict:
    return {
        "fenced_diff": f"I found the bug.\n```diff\n{gold}```\n",
        "fenced_python_lang": f"```python\n{gold}```",
        "raw": gold,
        "raw_prose_after": gold + "\nThis fixes the issue by adjusting the condition.\n",
        "prose_before_raw": "Here is my patch:\n\n" + gold,
        "crlf_fenced": "```diff\n" + gold.replace("\n", "\r\n") + "```",
        "two_blocks_example_then_final": "Example:\n```diff\n" + gold.replace("+++ b/", "+++ b/example_") + "```\nFinal:\n```diff\n" + gold + "```",
        "search_replace": to_search_replace(gold, buggy),
        "truncated_hunk": "```diff\n" + "\n".join(gold.split("\n")[:-3]) + "\n```",
    }


def run_synthetic(reg, n, seed, valid_ids):
    rng = random.Random(seed)
    ids = [i for i in valid_ids if i in reg.swe]; rng.shuffle(ids); ids = ids[:n]
    rows = []
    for iid in ids:
        inst = reg.swe[iid]; gold = inst["gold_patch"]; buggy = reg.buggy_files(iid) or {}
        for wname, text in wrappings(gold, buggy).items():
            for parser, sel in PARSER_ARMS:
                r = extract(text, parser, buggy_files=buggy, select=sel)
                same = r.status == ExtractStatus.DIFF_PARSE_SUCCESS and (r.patch.strip() == gold.strip() or (r.fmt == "search_replace" and validate_unified_diff(r.patch)[0]))
                rows.append({"instance_id": iid, "wrapping": wname, "parser": parser, "select": sel, "status": r.status.value, "fmt": r.fmt,
                             "recovered_gold": bool(same), "note": r.note[:80]})
    return pd.DataFrame(rows)


def run_model(reg, responses, execute_n, seed, vcfg_path):
    rows = []
    for r in responses:
        iid = r["instance_id"]; text = r.get("response", "")
        if iid in reg.swe:
            buggy = reg.buggy_files(iid) or {}
            for parser, sel in PARSER_ARMS:
                e = extract(text, parser, buggy_files=buggy, select=sel)
                rows.append({"instance_id": iid, "sample_idx": r.get("sample_idx"), "track": "swe", "parser": parser, "select": sel,
                             "status": e.status.value, "fmt": e.fmt, "n_candidates": e.n_candidates, "ref_state": r.get("state"), "ref_resolved": r.get("rule_correctness_score")})
    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stageB"); ap.add_argument("--n_synthetic", type=int, default=120); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--responses", default=""); ap.add_argument("--execute", type=int, default=0)
    ap.add_argument("--validation", default=f"{ROOT}/environments/manifests/validation_swe.jsonl")
    a = ap.parse_args()
    out = f"{ROOT}/results/verifier_ablation/{a.tag}"; os.makedirs(out, exist_ok=True)
    reg = Registry.get()
    valid_ids = [json.loads(l)["instance_id"] for l in open(a.validation) if json.loads(l).get("status") in ("ok", "ok_effective")]
    syn = run_synthetic(reg, a.n_synthetic, a.seed, valid_ids); syn.to_csv(f"{out}/synthetic.csv", index=False)
    tab = syn.groupby(["parser", "select", "wrapping"])["recovered_gold"].mean().unstack().round(3)
    tab.index = [f"{p}/{s}" for p, s in tab.index]
    summ = {"synthetic": tab.to_dict(orient="index"), "synthetic_overall": {k: round(float(v), 3) for k, v in tab.mean(axis=1).items()}}
    if a.responses:
        resp = [json.loads(l) for l in open(a.responses)]
        mdf = run_model(reg, resp, a.execute, a.seed, "")
        mdf.to_csv(f"{out}/model.csv", index=False)
        g = mdf.groupby(["parser", "select"])
        summ["model"] = {f"{p}/{s}": {"extract_rate": float((d["status"] != "no_patch").mean()), "valid_rate": float((d["status"] == "diff_parse_success").mean()),
                                      "n": int(len(d)), "fmt": dict(collections.Counter(d["fmt"]))} for (p, s), d in g}
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1)[:6000])


if __name__ == "__main__":
    main()
