"""Config-driven reward composition. Every component is logged; the final formula depends on
RewardConfig. Two families:

  ladder   (baseline family; rl_code_v1 `score_of` is the default parameterisation)
           gates: no_patch -> parse_fail -> invalid -> apply_fail -> ran
           ran:   frac==1 & no regression -> s_resolved
                  frac==1 & regression    -> max(floor, s_f2p_all * M)
                  else                    -> max(floor, (s_applies + w_partial * T(frac)) * M)
  additive (non-gating candidate) final = clip(sum_i w_i * c_i - penalties, 0, 1)

T = f2p.transform, M = p2p.multiplier. rubric_score (from rubric.py) enters with rubric_weight in
both families (ladder: added to the ran-branch base before the P2P multiplier when rusca_stage
is 'scaffold'/'transition'; weight 0 in 'rule').
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

from . import f2p as F2P
from . import p2p as P2P
from .schemas import ApplyStatus, ErrKind, ExecutionResult, ExtractStatus, ExtractionResult, RewardComponents


@dataclass
class RewardConfig:
    version: str = "rw_v001_baseline"
    mode: str = "ladder"                 # ladder | additive
    gating: bool = True                  # additive family only: False -> format/rubric credit even if apply fails
    # gate scores (ladder) -----------------------------------------------------------------
    s_no_patch: float = 0.0
    s_parse_fail: float = 0.0
    s_invalid: float = 0.0
    s_apply_fail: float = 0.05
    s_applies: float = 0.10
    w_partial: float = 0.60
    s_f2p_all_regressed: float = 0.85
    s_resolved: float = 1.0
    floor_after_p2p: float = 0.05
    # F2P transform ------------------------------------------------------------------------
    f2p_mode: str = "pow"                # binary | linear | sqrt | pow | ladder
    f2p_gamma: float = 1.5
    f2p_ladder: tuple = ((0.34, 0.2), (0.67, 0.5), (1.0, 1.0))
    # P2P penalty --------------------------------------------------------------------------
    p2p_mode: str = "pow"                # none | linear | pow | capped | tolerance | confidence | catastrophic
    p2p_max: float = 0.75
    p2p_gamma: float = 1.5
    p2p_tolerance: int = 0
    p2p_cap: float = 0.5
    p2p_conf_z: float = 1.0
    p2p_catastrophic_frac: float = 0.5
    p2p_catastrophic_mult: float = 0.25
    # additive weights ---------------------------------------------------------------------
    patch_format_weight: float = 0.05
    patch_apply_weight: float = 0.05
    f2p_weight: float = 0.60
    p2p_weight: float = 0.0              # additive credit for preserved P2P (usually 0; penalty below)
    regression_penalty: float = 0.5      # additive: penalty * (1 - M)
    resolved_bonus: float = 0.30
    timeout_penalty: float = 0.0
    # rubric / RUSCA -----------------------------------------------------------------------
    rubric_weight: float = 0.0
    rusca_stage: str = "rule"            # scaffold | transition | rule
    # length -------------------------------------------------------------------------------
    overlong_buffer: int = 0
    overlong_penalty: float = 1.0
    max_response_length: int = 4096
    # infra --------------------------------------------------------------------------------
    infra_score: float = 0.0
    notes: str = ""

    @classmethod
    def load(cls, path: str) -> "RewardConfig":
        import yaml
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        d.pop("_comment", None)
        if "f2p_ladder" in d:
            d["f2p_ladder"] = tuple(tuple(x) for x in d["f2p_ladder"])
        return cls(**d)

    def dump(self) -> dict:
        d = asdict(self)
        d["f2p_ladder"] = [list(x) for x in self.f2p_ladder]
        return d


def _state(ext: ExtractionResult, er: ExecutionResult) -> str:
    if ext.status == ExtractStatus.NO_PATCH:
        return "no_patch"
    if ext.status == ExtractStatus.DIFF_PARSE_FAIL:
        return "parse_fail"
    if er.err_kind.is_invalid:
        return "invalid"
    if er.err_kind.is_infra:
        return "infra"
    if er.err_kind == ErrKind.FORMAT_APPLY_FAIL or er.apply.status == ApplyStatus.PATCH_APPLY_FAIL:
        return "apply_fail"
    if er.ran:
        return "ran"
    return "infra"


def overlong(score: float, response_len: Optional[int], cfg: RewardConfig) -> tuple[float, float]:
    """DAPO-style soft overlong penalty (port of t15_code_reward._overlong). Returns (penalty, fraction)."""
    if response_len is None or cfg.overlong_buffer <= 0 or cfg.max_response_length <= 0:
        return 0.0, 0.0
    exceed = int(response_len) - (cfg.max_response_length - cfg.overlong_buffer)
    if exceed <= 0:
        return 0.0, 0.0
    frac = min(1.0, exceed / cfg.overlong_buffer)
    return -frac * cfg.overlong_penalty, frac


def compute_reward(ext: ExtractionResult, er: ExecutionResult, cfg: RewardConfig, *, rubric_score: float = 0.0,
                   response_len: Optional[int] = None, track: str = "", extra_flags=()) -> RewardComponents:
    rc = RewardComponents(track=track, reward_version=cfg.version, patch_fmt=ext.fmt, parser=ext.parser)
    st = _state(ext, er)
    rc.state = st
    rc.err_kind = er.err_kind.value if st not in ("no_patch", "parse_fail") else ("format_no_patch" if st == "no_patch" else "format_parse_fail")
    rc.patch_extraction_score = 0.0 if ext.status == ExtractStatus.NO_PATCH else 1.0
    rc.patch_format_score = 1.0 if ext.status == ExtractStatus.DIFF_PARSE_SUCCESS else 0.0
    rc.patch_apply_score = 1.0 if (st == "ran" or er.apply.status in (ApplyStatus.PATCH_APPLY_SUCCESS, ApplyStatus.GIT_APPLY_CHECK_SUCCESS)) else 0.0
    rc.n_f2p, rc.t_f2p = float(er.f2p.n_passed), float(er.f2p.total)
    rc.n_p2p, rc.t_p2p = float(er.p2p.n_passed), float(er.p2p.total)
    rc.touched_files = float(len(ext.touched_files))
    rc.hack_flags = ";".join(list(er.hack_flags) + list(extra_flags))
    rc.rubric_score = float(rubric_score or 0.0)
    rc.timeout_error_penalty = -cfg.timeout_penalty if er.timed_out else 0.0
    rc.infra_excluded = 1.0 if st == "infra" else 0.0

    frac = er.f2p_frac if st == "ran" else 0.0
    p2p_frac = er.p2p_frac if st == "ran" else 1.0
    T = F2P.transform(frac, cfg.f2p_mode, cfg.f2p_gamma, cfg.f2p_ladder) if st == "ran" else 0.0
    M = P2P.multiplier(er.p2p.n_passed, er.p2p.total, cfg.p2p_mode, cfg.p2p_max, cfg.p2p_gamma, cfg.p2p_tolerance,
                       cfg.p2p_cap, cfg.p2p_conf_z, cfg.p2p_catastrophic_frac, cfg.p2p_catastrophic_mult) if st == "ran" else 1.0
    rc.f2p_score, rc.f2p_frac, rc.p2p_preservation_score, rc.p2p_frac = T, frac, M, p2p_frac
    resolved = bool(st == "ran" and er.resolved)
    rc.rule_correctness_score = 1.0 if resolved else 0.0
    rc.answer_match = 1.0 if resolved else 0.0
    rw = cfg.rubric_weight if cfg.rusca_stage != "rule" else 0.0

    if st == "infra":
        base = cfg.infra_score
    elif cfg.mode == "ladder":
        if st == "no_patch":
            base = cfg.s_no_patch
        elif st == "parse_fail":
            base = cfg.s_parse_fail
        elif st == "invalid":
            base = cfg.s_invalid
        elif st == "apply_fail":
            base = cfg.s_apply_fail
        else:
            reg = 1.0 - p2p_frac
            if frac >= 1.0 and reg <= 0.0:
                base = cfg.s_resolved
            elif frac >= 1.0:
                base = max(cfg.floor_after_p2p, cfg.s_f2p_all_regressed * M)
            else:
                base = max(cfg.floor_after_p2p, (cfg.s_applies + cfg.w_partial * T + rw * rc.rubric_score) * M)
            if rw > 0 and frac >= 1.0:
                base = min(1.0, base + rw * rc.rubric_score * (1.0 if reg <= 0 else M))
    else:  # additive
        if st == "invalid":
            base = cfg.s_invalid
        else:
            gate_ok = st == "ran" or not cfg.gating
            base = cfg.patch_format_weight * rc.patch_format_score + cfg.patch_apply_weight * rc.patch_apply_score
            if gate_ok:
                base += rw * rc.rubric_score
            if st == "ran":
                base += cfg.f2p_weight * T + cfg.p2p_weight * p2p_frac + (cfg.resolved_bonus if resolved else 0.0)
                base -= cfg.regression_penalty * (1.0 - M)
            base = min(1.0, max(0.0, base))
    base = float(base) + rc.timeout_error_penalty
    rc.score_before_overlong = base
    pen, _ = overlong(base, response_len, cfg)
    rc.length_component = pen
    rc.final_reward = base + pen
    rc.gated_out = 1.0 if st in ("no_patch", "parse_fail", "invalid", "apply_fail") else 0.0
    return rc


def baseline_score_of(state: str, f2p_frac: float, p2p_frac: float) -> float:
    """Reference port of rl_code_v1/reward/t15_code_judge.score_of for regression tests."""
    if state == "no_patch":
        return 0.0
    if state == "apply_fail":
        return 0.05
    if state == "invalid":
        return 0.0
    reg = max(0.0, 1.0 - p2p_frac)
    mult = 1.0 - 0.75 * (reg ** 1.5)
    if f2p_frac >= 1.0:
        return 1.0 if reg <= 0 else max(0.05, 0.85 * mult)
    return max(0.05, (0.10 + 0.60 * (max(0.0, f2p_frac) ** 1.5)) * mult)
