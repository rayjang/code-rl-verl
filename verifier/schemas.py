"""Typed records shared by every verifier stage.

Every stage returns a dataclass; `to_flat()` turns the tree into a flat dict of float/int/str
(no None/NaN/collections) so it can be logged by verl's reward manager (contract inherited
from the T01 math scorer: same key set on every row, float/str values only, deterministic).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from typing import Any, Optional


class ExtractStatus(str, Enum):
    NO_PATCH = "no_patch"                      # nothing that looks like a patch / code block
    DIFF_PARSE_SUCCESS = "diff_parse_success"  # a structurally valid patch was extracted
    DIFF_PARSE_FAIL = "diff_parse_fail"        # patch-like text found but structurally invalid


class ApplyStatus(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    GIT_APPLY_CHECK_SUCCESS = "git_apply_check_success"
    PATCH_APPLY_SUCCESS = "patch_apply_success"
    PATCH_APPLY_FAIL = "patch_apply_fail"


class ErrKind(str, Enum):
    OK = "ok"
    # format / structural failures (model fault, learnable)
    FORMAT_NO_PATCH = "format_no_patch"
    FORMAT_PARSE_FAIL = "format_parse_fail"
    FORMAT_APPLY_FAIL = "format_apply_fail"
    # rule violations (model fault, reward 0, NOT excluded)
    INVALID_EDITS_TESTS = "invalid_edits_tests"
    INVALID_FORBIDDEN_PATH = "invalid_forbidden_path"
    INVALID_ALWAYS_TRUE_EQ = "invalid_always_true_eq"
    INVALID_HACK_PATTERN = "invalid_hack_pattern"
    # infrastructure failures (not model fault -> infra_excluded=1)
    INFRA_TIMEOUT = "infra_timeout"
    INFRA_ENV = "infra_env"
    INFRA_DATA = "infra_data"
    INFRA_OTHER = "infra_other"

    @property
    def is_infra(self) -> bool:
        return self.value.startswith("infra_")

    @property
    def is_invalid(self) -> bool:
        return self.value.startswith("invalid_")

    @property
    def is_format(self) -> bool:
        return self.value.startswith("format_")


@dataclass
class ExtractionResult:
    status: ExtractStatus
    patch: str = ""                # normalised unified diff ("" if none)
    fmt: str = "none"              # which extraction path fired (fenced_diff, raw_git, search_replace, ...)
    parser: str = ""               # parser candidate name
    n_candidates: int = 0          # how many patch-like blocks were seen
    note: str = ""                 # human-readable reason on failure
    touched_files: tuple = ()      # paths named in the diff headers

    @property
    def ok(self) -> bool:
        return self.status == ExtractStatus.DIFF_PARSE_SUCCESS and bool(self.patch.strip())


@dataclass
class ApplyResult:
    status: ApplyStatus
    touched_files: tuple = ()
    error: str = ""
    stderr_tail: str = ""


@dataclass
class TestOutcome:
    passed: tuple = ()
    failed: tuple = ()
    missing: tuple = ()   # expected test id not present in the pytest summary at all

    @property
    def total(self) -> int:
        return len(self.passed) + len(self.failed) + len(self.missing)

    @property
    def n_passed(self) -> int:
        return len(self.passed)

    @property
    def frac(self) -> float:
        return (self.n_passed / self.total) if self.total else 0.0


@dataclass
class ExecutionResult:
    ran: bool
    err_kind: ErrKind = ErrKind.OK
    err: str = ""
    f2p: TestOutcome = field(default_factory=TestOutcome)
    p2p: TestOutcome = field(default_factory=TestOutcome)
    runtime_s: float = 0.0
    timed_out: bool = False
    log_tail: str = ""
    apply: ApplyResult = field(default_factory=lambda: ApplyResult(ApplyStatus.NOT_ATTEMPTED))
    hack_flags: tuple = ()

    @property
    def resolved(self) -> bool:
        return self.ran and self.f2p.total > 0 and self.f2p.n_passed == self.f2p.total and self.p2p.n_passed == self.p2p.total

    @property
    def f2p_frac(self) -> float:
        return self.f2p.frac

    @property
    def p2p_frac(self) -> float:
        return self.p2p.frac if self.p2p.total else 1.0


@dataclass
class RewardComponents:
    """Every component is logged even when it does not enter the final formula."""
    patch_extraction_score: float = 0.0   # 1 if a patch/code block was extracted
    patch_format_score: float = 0.0       # 1 if extracted patch is structurally valid
    patch_apply_score: float = 0.0        # 1 if it applies to the base checkout
    f2p_score: float = 0.0                # transformed F2P credit in [0,1]
    f2p_frac: float = 0.0                 # raw passed/total
    p2p_preservation_score: float = 1.0   # multiplicative factor in [0,1] (1 = no regression)
    p2p_frac: float = 1.0
    rubric_score: float = 0.0             # machine-checkable rubric satisfaction in [0,1]
    rule_correctness_score: float = 0.0   # 1 if resolved (all F2P and no observed regression)
    timeout_error_penalty: float = 0.0
    length_component: float = 0.0
    final_reward: float = 0.0
    score_before_overlong: float = 0.0
    answer_match: float = 0.0             # 1 only when resolved (DDCA / T01 contract)
    infra_excluded: float = 0.0
    gated_out: float = 0.0                # 1 if a gate stopped the ladder
    err_kind: str = "ok"
    state: str = "no_patch"
    patch_fmt: str = "none"
    parser: str = ""
    track: str = ""
    n_f2p: float = 0.0
    t_f2p: float = 0.0
    n_p2p: float = 0.0
    t_p2p: float = 0.0
    touched_files: float = 0.0
    hack_flags: str = ""
    reward_version: str = ""
    verifier_version: str = ""
    rubric_version: str = ""


def _clean(v: Any) -> Any:
    """float/int/str only; None/NaN/inf/bool/collections are coerced (contract invariant 2)."""
    if isinstance(v, Enum):
        v = v.value
    if isinstance(v, bool):
        return float(v)
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple, set)):
        return ";".join(str(x) for x in v)
    return str(v)


def to_flat(obj: Any, prefix: str = "") -> dict:
    """Flatten a dataclass (recursively) into {key: float|str}."""
    out: dict = {}
    if hasattr(obj, "__dataclass_fields__"):
        for f in fields(obj):
            out.update(to_flat(getattr(obj, f.name), f"{prefix}{f.name}."))
        # expose read-only properties that matter
        for prop in ("resolved", "f2p_frac", "p2p_frac", "ok", "total", "n_passed", "frac"):
            if hasattr(type(obj), prop) and isinstance(getattr(type(obj), prop), property):
                try:
                    out[f"{prefix}{prop}"] = _clean(getattr(obj, prop))
                except Exception:
                    pass
        return out
    key = prefix[:-1] if prefix.endswith(".") else prefix
    out[key] = _clean(obj)
    return out


def components_to_dict(rc: RewardComponents) -> dict:
    d = {k: _clean(v) for k, v in asdict(rc).items()}
    d["score"] = d["final_reward"]
    return d
