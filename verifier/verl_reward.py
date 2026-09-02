"""verl entry point:  reward.custom_reward_function.path=verifier/verl_reward.py  name=compute_score

Environment (all optional):
  RL_INDEX_DIR        index directory (default sources/rl_code_v1/data/index)
  RL_SIF_DIR          image directory (default environments/sif)
  VERIFIER_CONFIG     yaml with verifier settings (parser, select, apply_mode, p2p_cap, isolation ...)
  REWARD_CONFIG       yaml RewardConfig (default configs/reward/rw_v001_baseline.yaml)
  RUBRIC_CONFIG       yaml with rubric settings (requested_fmt, max_extra_ratio)
  RUSCA_STEP_FILE     file whose content is the current training step (written by the trainer/agent loop)
  VERIFIER_RUN_DIR    scratch dir for containers (default /tmp/<user>_verifier_run)
  VERIFIER_CONCURRENCY max concurrent container executions in this process (default 8)
  VERIFIER_LOG_JSONL  if set, one JSON line per scored sample is appended (instance-level results)

Contract (inherited from T01/T15): every call returns the SAME key set, values float/str only,
deterministic. infra failures return infra_excluded=1 and score=infra_score.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys  # noqa: E402
if ROOT not in sys.path:      # verl loads this file by path (not as a package) -> absolute imports
    sys.path.insert(0, ROOT)

from verifier.anti_hacking import inspect_patch  # noqa: E402
from verifier.diff_parser import extract, extract_code_block, touched_paths  # noqa: E402
from verifier.registry import Registry  # noqa: E402
from verifier.reward import RewardConfig, compute_reward  # noqa: E402
from verifier.rubric import evaluate_rubric, parse_rubrics  # noqa: E402
from verifier.rusca import RuscaConfig, stage as rusca_stage  # noqa: E402
from verifier.schemas import (ApplyResult, ApplyStatus, ErrKind, ExecutionResult, ExtractStatus, ExtractionResult,  # noqa: E402
                              RewardComponents, components_to_dict)
from verifier.swe_runner import SweSmithRunner  # noqa: E402
from verifier.ut_runner import UnitTestRunner  # noqa: E402
_USER = os.environ.get("USER", "user")


def _yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class VerifierService:
    """Process-wide singleton holding runners + configs (verl reward workers are long-lived)."""
    _inst = None
    _lock = threading.Lock()

    def __init__(self):
        vcfg_path = os.environ.get("VERIFIER_CONFIG", os.path.join(ROOT, "configs/verifier/vf_v001.yaml"))
        self.vcfg = _yaml(vcfg_path) if os.path.exists(vcfg_path) else {}
        rcfg_path = os.environ.get("REWARD_CONFIG", os.path.join(ROOT, "configs/reward/rw_v001_baseline.yaml"))
        self.rcfg = RewardConfig.load(rcfg_path) if os.path.exists(rcfg_path) else RewardConfig()
        rub_path = os.environ.get("RUBRIC_CONFIG", "")
        self.rubcfg = _yaml(rub_path) if rub_path and os.path.exists(rub_path) else {}
        self.rusca = RuscaConfig(**(self.vcfg.get("rusca") or {})) if self.vcfg.get("rusca") is not None else RuscaConfig(enable=False)
        self.reg = Registry.get()
        run_dir = os.environ.get("VERIFIER_RUN_DIR", f"/tmp/{_USER}_verifier_run")
        sif_dir = self.reg.sif_dir
        self.swe = SweSmithRunner(sif_dir, run_dir, os.path.join(ROOT, "environments/cache_v4"),
                                  timeout=int(self.vcfg.get("swe_timeout", 900)), p2p_cap=int(self.vcfg.get("p2p_cap", 30)),
                                  apply_mode=self.vcfg.get("apply_mode", "strict"), isolate=bool(self.vcfg.get("isolate", True)),
                                  ignore_whitespace=bool(self.vcfg.get("ignore_whitespace", False)),
                                  use_p2p_effective=bool(self.vcfg.get("use_effective_tests", True)))
        self.ut = UnitTestRunner(os.path.join(sif_dir, "python_3.11-slim-bookworm.sif"), os.path.join(ROOT, "environments/ut_venv"),
                                 run_dir, isolate=bool(self.vcfg.get("isolate", True)), canary=bool(self.vcfg.get("canary", True)))
        self.sem = threading.Semaphore(int(os.environ.get("VERIFIER_CONCURRENCY", "8")))
        self.log_path = os.environ.get("VERIFIER_LOG_JSONL", "")
        self.log_lock = threading.Lock()
        self.version = f"{self.vcfg.get('version', 'vf_v001')}"
        self.rubric_version = self.rubcfg.get("version", "rb_v000_none")

    @classmethod
    def get(cls) -> "VerifierService":
        with cls._lock:
            if cls._inst is None:
                cls._inst = VerifierService()
            return cls._inst

    # ------------------------------------------------------------------ step / stage
    def current_step(self) -> int:
        p = os.environ.get("RUSCA_STEP_FILE", "")
        if p and os.path.exists(p):
            try:
                return int(open(p).read().strip() or 0)
            except Exception:
                return 0
        return 0

    # ------------------------------------------------------------------ scoring
    def score(self, data_source: str, solution_str: str, ground_truth, extra_info: dict | None) -> dict:
        ei = dict(extra_info or {})
        iid = str(ei.get("instance_id") or ground_truth or "")
        track = self.reg.track_of(iid, str(data_source or ""))
        step = int(ei.get("global_step", self.current_step()))
        rcfg = self.rcfg
        if self.rusca.enable:
            rcfg = RewardConfig(**{**asdict(rcfg), "rusca_stage": rusca_stage(step, self.rusca)})
        parser = self.vcfg.get("parser", "fallback")
        select = self.vcfg.get("select", "last")
        resp_len = ei.get("valid_response_length")
        t0 = time.time()
        flags: tuple = ()
        rub_score = 0.0
        if track == "unittest":
            inst = self.reg.ut.get(iid)
            if inst is None:
                ext, er = ExtractionResult(ExtractStatus.NO_PATCH, "", "none", "code_block"), ExecutionResult(False, ErrKind.INFRA_ENV, "unknown_instance")
            else:
                ext = extract_code_block(solution_str, select=select)
                if ext.status == ExtractStatus.DIFF_PARSE_SUCCESS:
                    with self.sem:
                        er = self.ut.run(inst, ext.patch)
                else:
                    er = ExecutionResult(False, ErrKind.FORMAT_NO_PATCH if ext.status == ExtractStatus.NO_PATCH else ErrKind.FORMAT_PARSE_FAIL, ext.note)
            rubrics = parse_rubrics(ei.get("rubrics"))
            rub_score = evaluate_rubric(rubrics, ext, er, requested_fmt="any").score if rubrics else 0.0
        elif track == "swe":
            inst = self.reg.swe.get(iid)
            if inst is None:
                ext, er = ExtractionResult(ExtractStatus.NO_PATCH, "", "none", parser), ExecutionResult(False, ErrKind.INFRA_ENV, "unknown_instance")
            else:
                ext = extract(solution_str, parser, buggy_files=self.reg.buggy_files(iid), select=select,
                              allow_search_replace=bool(self.vcfg.get("allow_search_replace", True)))
                if ext.status != ExtractStatus.DIFF_PARSE_SUCCESS:
                    er = ExecutionResult(False, ErrKind.FORMAT_NO_PATCH if ext.status == ExtractStatus.NO_PATCH else ErrKind.FORMAT_PARSE_FAIL, ext.note)
                else:
                    flags, verdict = inspect_patch(ext.patch, inst, forbid_tests=bool(self.vcfg.get("forbid_test_edits", True)),
                                                   forbid_config=bool(self.vcfg.get("forbid_config_edits", True)))
                    if verdict == "invalid":
                        kind = ErrKind.INVALID_EDITS_TESTS if "edits_tests" in flags else ErrKind.INVALID_FORBIDDEN_PATH if ("path_escape" in flags or "edits_verifier" in flags or "edits_config" in flags) else ErrKind.INVALID_HACK_PATTERN
                        er = ExecutionResult(False, kind, "patch_inspection:" + ",".join(flags), hack_flags=flags)
                    else:
                        with self.sem:
                            er = self.swe.run(inst, ext.patch)
                        er.hack_flags = flags
                rubrics = parse_rubrics(ei.get("rubrics"))
                if rubrics:
                    gold = inst.get("gold_patch") or ""
                    rub_score = evaluate_rubric(rubrics, ext, er, gold_touched=touched_paths(gold), gold_patch_len=len(gold),
                                                requested_fmt=self.vcfg.get("requested_fmt", "fenced_diff")).score
        else:
            ext, er = ExtractionResult(ExtractStatus.NO_PATCH, "", "none", parser), ExecutionResult(False, ErrKind.INFRA_DATA, f"unknown track for {data_source}")
        rc = compute_reward(ext, er, rcfg, rubric_score=rub_score, response_len=resp_len, track=track, extra_flags=flags)
        rc.verifier_version, rc.rubric_version = self.version, self.rubric_version
        out = components_to_dict(rc)
        out["verify_runtime_s"] = round(time.time() - t0, 2)
        out["rusca_step"] = float(step)
        if self.log_path:
            rec = {"instance_id": iid, "data_source": str(data_source), "step": step, "ts": time.time(), **out,
                   "f2p_failed": ";".join(er.f2p.failed[:20]), "p2p_failed": ";".join(er.p2p.failed[:20]),
                   "note": ext.note[:200], "err": er.err[:200], "resp_len": resp_len}
            with self.log_lock, open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return out


_TEMPLATE_KEYS = None


def compute_score_dict(data_source, solution_str, ground_truth, extra_info=None, **kw) -> dict:
    global _TEMPLATE_KEYS
    svc = VerifierService.get()
    try:
        out = svc.score(data_source, solution_str, ground_truth, extra_info)
    except Exception as e:  # never crash the trainer; surface as infra
        rc = RewardComponents(err_kind="infra_other", state="infra", infra_excluded=1.0, reward_version=svc.rcfg.version,
                              verifier_version=svc.version, rubric_version=svc.rubric_version, hack_flags=f"exception:{type(e).__name__}")
        out = components_to_dict(rc)
        out["verify_runtime_s"] = 0.0
        out["rusca_step"] = 0.0
    if _TEMPLATE_KEYS is None:
        _TEMPLATE_KEYS = tuple(out.keys())
    return {k: out.get(k, 0.0 if not isinstance(out.get(k), str) else "") for k in _TEMPLATE_KEYS}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kw) -> dict:
    return compute_score_dict(data_source, solution_str, ground_truth, extra_info, **kw)
