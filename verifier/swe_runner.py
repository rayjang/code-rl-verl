"""SWE-smith execution runner (repository image + instance branch), singularity based.

Flow per evaluation (mirrors rl_code_v1/reward/swe_exec_v2.py, with three changes):
  1. base tree of the instance = `git archive origin/<iid>` (buggy state) + test files restored from
     `main` (tests are deleted on the instance branch). Extracted ONCE per instance inside the
     container and cached as a tarball (cache/<iid>.tar) -> later rollouts just untar.
  2. the model patch is applied on the HOST with `git apply` (strict, no fuzz) so that
     patch_apply_fail is decided before any container is launched. (baseline: git apply -> patch --fuzz=5)
  3. pytest runs inside the image with --containall --net --network=none --writable-tmpfs and the
     patched tree bound over /testbed. Only F2P + capped P2P ids are run.
Everything returns ExecutionResult; nothing here computes a reward.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Optional

from .patch_apply import apply_patch
from .pytest_parse import clean_ids, evaluate, parse_summary, pytest_ran
from .schemas import ApplyResult, ApplyStatus, ErrKind, ExecutionResult, TestOutcome

SING = os.environ.get("SINGULARITY_BIN") or shutil.which("apptainer") or shutil.which("singularity") \
    or "/apps/applications/singularity/4.3.4/bin/singularity"


class SweSmithRunner:
    def __init__(self, sif_dir: str, run_dir: str, cache_dir: str, *, timeout: int = 900, extract_timeout: int = 300,
                 p2p_cap: int = 30, apply_mode: str = "strict", isolate: bool = True, keep_workdir: bool = False,
                 singularity_bin: str = SING):
        self.sif_dir, self.run_dir, self.cache_dir = sif_dir, run_dir, cache_dir
        self.timeout, self.extract_timeout, self.p2p_cap = timeout, extract_timeout, p2p_cap
        self.apply_mode, self.isolate, self.keep = apply_mode, isolate, keep_workdir
        self.sing = singularity_bin
        self._locks: dict = {}
        self._lock = threading.Lock()
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def sif_for(self, inst: dict) -> str:
        base = inst["image_name"].rsplit("/", 1)[-1].replace(":", "_") + ".sif"
        return os.path.join(self.sif_dir, base)

    def _iso_flags(self) -> list:
        return ["--containall", "--no-home", "--cleanenv", "--net", "--network=none"] if self.isolate else []

    def test_ids(self, inst: dict) -> tuple[list, list]:
        f2p = inst.get("FAIL_TO_PASS") or []
        p2p = inst.get("PASS_TO_PASS") or []
        if isinstance(f2p, str):
            f2p = json.loads(f2p)
        if isinstance(p2p, str):
            p2p = json.loads(p2p)
        return clean_ids(f2p), clean_ids(p2p)[: self.p2p_cap]

    def _inst_lock(self, iid: str) -> threading.Lock:
        with self._lock:
            return self._locks.setdefault(iid, threading.Lock())

    # ------------------------------------------------------------------ base tree cache
    def base_tar(self, inst: dict) -> tuple[Optional[str], str]:
        """Return (tar_path, err). Builds the cache on first use."""
        iid = inst["instance_id"]
        tar = os.path.join(self.cache_dir, iid + ".tar")
        with self._inst_lock(iid):
            if os.path.exists(tar) and os.path.getsize(tar) > 0:
                return tar, ""
            sif = self.sif_for(inst)
            if not os.path.exists(sif):
                return None, f"sif missing: {sif}"
            f2p, p2p = self.test_ids(inst)
            test_files = sorted({t.split("::")[0] for t in f2p + p2p})
            wd = tempfile.mkdtemp(dir=self.run_dir, prefix="extract." + iid[:40] + ".")
            os.makedirs(f"{wd}/tb", exist_ok=True)
            tf_args = " ".join(shlex.quote(t) for t in test_files)
            script = (f"cd /testbed && git -c safe.directory='*' archive origin/{shlex.quote(iid)} | tar -x -C /wd/tb && "
                      f"git -c safe.directory='*' archive main -- {tf_args} | tar -x -C /wd/tb && echo EXTRACT_OK")
            try:
                r = subprocess.run([self.sing, "exec", *self._iso_flags(), "--bind", f"{wd}:/wd", sif, "bash", "-c", script],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=self.extract_timeout)
            except subprocess.TimeoutExpired:
                shutil.rmtree(wd, ignore_errors=True)
                return None, "extract_timeout"
            if r.returncode != 0 or "EXTRACT_OK" not in r.stdout:
                err = (r.stderr or r.stdout)[-400:]
                shutil.rmtree(wd, ignore_errors=True)
                return None, f"branch_extract_fail: {err}"
            missing = [t for t in test_files if not os.path.exists(os.path.join(wd, "tb", t))]
            if missing:
                shutil.rmtree(wd, ignore_errors=True)
                return None, f"test files missing after restore: {missing[:3]}"
            tmp = tar + ".tmp"
            subprocess.run(["tar", "-cf", tmp, "-C", wd, "tb"], check=True)
            os.replace(tmp, tar)
            shutil.rmtree(wd, ignore_errors=True)
            return tar, ""

    # ------------------------------------------------------------------ evaluation
    def run(self, inst: dict, patch: str, *, reverse: bool = False, extra_env: Optional[dict] = None) -> ExecutionResult:
        t0 = time.time()
        iid = inst["instance_id"]
        f2p, p2p = self.test_ids(inst)
        if not f2p:
            return ExecutionResult(False, ErrKind.INFRA_DATA, "no_valid_f2p")
        sif = self.sif_for(inst)
        if not os.path.exists(sif):
            return ExecutionResult(False, ErrKind.INFRA_ENV, f"sif missing: {sif}")
        tar, err = self.base_tar(inst)
        if tar is None:
            kind = ErrKind.INFRA_TIMEOUT if err == "extract_timeout" else ErrKind.INFRA_ENV
            return ExecutionResult(False, kind, err)
        wd = tempfile.mkdtemp(dir=self.run_dir, prefix=iid[:40] + ".")
        try:
            subprocess.run(["tar", "-xf", tar, "-C", wd], check=True)
            tree = os.path.join(wd, "tb")
            # ---- host-side apply (strict) ----
            if patch.strip():
                ap = apply_patch(patch, tree, mode=self.apply_mode, reverse=reverse)
            else:
                ap = ApplyResult(ApplyStatus.PATCH_APPLY_SUCCESS, (), "empty patch (no-op)")
            if ap.status == ApplyStatus.PATCH_APPLY_FAIL:
                return ExecutionResult(False, ErrKind.FORMAT_APPLY_FAIL, "patch_apply_fail", apply=ap,
                                       log_tail=ap.stderr_tail, runtime_s=time.time() - t0)
            tests = " ".join(shlex.quote(t) for t in (f2p + p2p))
            env_lines = "".join(f"export {k}={shlex.quote(str(v))}\n" for k, v in (extra_env or {}).items())
            runner = ("#!/bin/bash\n"
                      "source /opt/miniconda3/bin/activate testbed 2>/dev/null || export PATH=/opt/miniconda3/envs/testbed/bin:$PATH\n"
                      f"{env_lines}"
                      "export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-0}\n"
                      "cd /testbed\n"
                      f"timeout -s KILL {max(30, self.timeout - 15)} python -m pytest -rA --tb=no --color=no -p no:cacheprovider -q {tests} 2>&1\n"
                      "echo PYTEST_EXIT=$?\n")
            with open(f"{wd}/runner.sh", "w") as f:
                f.write(runner)
            cmd = [self.sing, "exec", *self._iso_flags(), "--writable-tmpfs", "--bind", f"{tree}:/testbed", "--bind", f"{wd}:/wd",
                   "--pwd", "/testbed", sif, "bash", "/wd/runner.sh"]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=self.timeout)
                out = (r.stdout or "") + (r.stderr or "")
            except subprocess.TimeoutExpired as e:
                out = ((e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))
                return ExecutionResult(False, ErrKind.INFRA_TIMEOUT, "timeout", apply=ap, timed_out=True,
                                       log_tail=out[-2000:], runtime_s=time.time() - t0)
            with open(f"{wd}/out.log", "w") as f:
                f.write(out)
            res = parse_summary(out)
            if not res and not pytest_ran(out):
                return ExecutionResult(False, ErrKind.INFRA_ENV, "pytest_did_not_run", apply=ap,
                                       log_tail=out[-2000:], runtime_s=time.time() - t0)
            f2p_o = evaluate(res, f2p)
            p2p_o = evaluate(res, p2p)
            return ExecutionResult(True, ErrKind.OK, "", f2p_o, p2p_o, runtime_s=time.time() - t0,
                                   timed_out=("PYTEST_EXIT=137" in out), log_tail=out[-2500:], apply=ap)
        finally:
            if not self.keep:
                shutil.rmtree(wd, ignore_errors=True)
