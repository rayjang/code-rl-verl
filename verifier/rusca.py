"""RUSCA (Rubric-Scaffolded RL) schedule and prompt scaffolding for the code profile.

What is documented locally (sources/rl_code_v1/docs/MATH_REWARD.md section 4, t01_math_reward_rubric README):
  * scaffolding criteria are injected into the rollout prompt only; the reward stays rule-based;
  * a step-dependent factor lam(step) drives how many criteria are injected: lam=1.0 at step 0,
    0.5 at step 10 of 50, 0.0067 at step 12 of 50 ("scaffold fully removed at ~24% of training");
  * criteria are dropped by scaffold_rank (lowest-rank hints first) -> code profile ranks:
      0 failing_test / impl_hint, 1 approach, 2 edge_cases / complexity, 3 tests_pass, 4 output_format / no_extra
  * variant-level (group-level) differentiation: rollouts in the same group receive slightly different
    numbers of criteria (observed "6 6 6 5 6 5").
The exact lam() functional form is UNKNOWN (the original T15 bundle is missing); the logistic below
is fitted to the three documented points and is exposed as a config so the ablation can vary it.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


@dataclass
class RuscaConfig:
    enable: bool = True
    total_steps: int = 50
    center_frac: float = 0.20      # lam = 0.5 at 20% of training  (10/50)
    steepness: float = 2.5         # per-step logistic slope at total_steps=50; scaled by 50/total_steps
    min_lam_cutoff: float = 0.01   # below this nothing is injected (rule-only)
    group_jitter: float = 0.15     # +-15% per-variant differentiation of the injected count
    max_criteria: int = 6
    profile: str = "code"
    stage_boundaries: tuple = (0.10, 0.24)   # <10% scaffold, <24% transition, else rule (for reward rusca_stage)


def lam(step: int, cfg: RuscaConfig) -> float:
    if not cfg.enable or cfg.total_steps <= 0:
        return 0.0
    k = cfg.steepness * (50.0 / cfg.total_steps)
    c = cfg.center_frac * cfg.total_steps
    return 1.0 / (1.0 + math.exp(k * (step - c)))


def stage(step: int, cfg: RuscaConfig) -> str:
    if not cfg.enable:
        return "rule"
    f = step / max(1, cfg.total_steps)
    if f < cfg.stage_boundaries[0]:
        return "scaffold"
    if f < cfg.stage_boundaries[1]:
        return "transition"
    return "rule"


def n_inject(step: int, n_criteria: int, cfg: RuscaConfig, variant_key: str = "") -> int:
    l = lam(step, cfg)
    if l < cfg.min_lam_cutoff:
        return 0
    base = l * min(n_criteria, cfg.max_criteria)
    if cfg.group_jitter > 0 and variant_key:
        h = int(hashlib.md5(variant_key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        base *= 1.0 + cfg.group_jitter * (2 * h - 1)
    return max(0, min(n_criteria, int(round(base))))


def select_criteria(rubrics: list[dict], k: int) -> list[dict]:
    """Keep the k criteria with the LOWEST scaffold_rank (rank 0 = strongest hint), stable order."""
    inject = [c for c in rubrics if c["tags"].get("inject", True)]
    ordered = sorted(inject, key=lambda c: (int(c["tags"].get("scaffold_rank", 9)), rubrics.index(c)))
    return ordered[:k]


def scaffold_text(criteria: list[dict], lang: str = "en") -> str:
    if not criteria:
        return ""
    head = "다음 지침을 따르세요:" if lang == "ko" else "Follow these guidelines:"
    return head + "\n" + "\n".join(f"- {c['criterion']}" for c in criteria) + "\n"


def inject(messages: list[dict], criteria: list[dict], lang: str = "en") -> list[dict]:
    """Return a new message list with the scaffold appended to the last user message."""
    txt = scaffold_text(criteria, lang)
    if not txt:
        return messages
    out = [dict(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i]["content"] = out[i]["content"].rstrip() + "\n\n" + txt
            break
    return out
