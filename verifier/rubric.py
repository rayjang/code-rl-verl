"""Rubric handling for the code profile.

Rubric records live in extra_info.rubrics (list of {criterion, points, tags{judge,inject,scaffold_rank,
scaffold_kind,verify,reward}}) — the schema shared with the T01 math rubric (rubrics/rubric_math.jsonl).
In rl_code_v1 every criterion is inject=True / reward=False (scaffold-only). This module adds a
machine-checkable evaluation for the `verify` kinds that can be decided from execution results, so a
rubric can enter the reward without an LLM judge:

  failing_test  -> F2P pass fraction            tests_pass   -> no observed P2P regression
  output_format -> requested output format used complexity   -> no per-test timeout
  no_extra      -> touched files subset of gold-touched files and diff not > k x gold size
  approach / edge_cases / impl_hint -> not checkable (judge=false) -> excluded from the denominator
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Optional

from .schemas import ExecutionResult, ExtractionResult, ExtractStatus

CHECKABLE = {"failing_test", "tests_pass", "output_format", "complexity", "no_extra"}


def parse_rubrics(raw) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    out = []
    for c in list(raw):
        c = dict(c) if hasattr(c, "keys") else c
        if not isinstance(c, dict):
            continue
        tags = dict(c.get("tags") or {})
        out.append({"criterion": str(c.get("criterion", "")), "points": float(c.get("points") or 0.0), "tags": tags})
    return out


@dataclass
class RubricEval:
    score: float            # weighted satisfaction over checkable criteria in [0,1]
    checked: int
    satisfied: float        # sum of points satisfied
    total_points: float
    details: dict


def evaluate_rubric(rubrics: list[dict], ext: ExtractionResult, er: ExecutionResult, *, gold_touched: Iterable[str] = (),
                    gold_patch_len: int = 0, requested_fmt: str = "fenced_diff", max_extra_ratio: float = 3.0) -> RubricEval:
    got, tot, det, n = 0.0, 0.0, {}, 0
    ran = er.ran
    gold_touched = set(gold_touched)
    for c in rubrics:
        v = (c["tags"].get("verify") or "").strip()
        if v not in CHECKABLE:
            continue
        pts = c["points"]
        if v == "failing_test":
            s = er.f2p_frac if ran else 0.0
        elif v == "tests_pass":
            s = 1.0 if (ran and er.p2p.n_passed == er.p2p.total) else 0.0
        elif v == "output_format":
            s = 1.0 if (ext.status == ExtractStatus.DIFF_PARSE_SUCCESS and (requested_fmt in ("any", ext.fmt))) else 0.0
        elif v == "complexity":
            s = 1.0 if (ran and not er.timed_out) else 0.0
        elif v == "no_extra":
            if not ext.touched_files:
                s = 0.0
            else:
                subset = set(ext.touched_files) <= gold_touched if gold_touched else True
                size_ok = (len(ext.patch) <= max_extra_ratio * max(1, gold_patch_len)) if gold_patch_len else True
                s = 1.0 if (subset and size_ok) else (0.5 if subset or size_ok else 0.0)
        else:
            continue
        got += pts * s
        tot += pts
        det[v] = round(s, 4)
        n += 1
    return RubricEval(score=(got / tot) if tot else 0.0, checked=n, satisfied=got, total_points=tot, details=det)
