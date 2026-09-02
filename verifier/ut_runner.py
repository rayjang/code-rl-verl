"""Unit-test track runner: one isolated container launch per rollout, driver inside.

Sandbox = python:3.11-slim .sif + host venv (pytest/numpy/sympy) bound read-only at /ut_venv.
Isolation flags: --containall --no-home --cleanenv --net --network=none --writable-tmpfs, only the
per-rollout work dir is bound (read-write) at /work.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional

from .schemas import ErrKind, ExecutionResult, TestOutcome

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "ut_driver.py")
SING = os.environ.get("SINGULARITY_BIN") or shutil.which("apptainer") or shutil.which("singularity") \
    or "/apps/applications/singularity/4.3.4/bin/singularity"


class UnitTestRunner:
    def __init__(self, sif: str, venv_dir: str, run_dir: str, *, isolate: bool = True, keep_workdir: bool = False,
                 singularity_bin: str = SING, max_container_s: int = 600, canary: bool = True,
                 local_python: Optional[str] = None):
        """local_python: if set, bypass the container and run the driver with that interpreter
        (NO isolation; unit tests / fixtures only)."""
        self.sif, self.venv, self.run_dir = sif, venv_dir, run_dir
        self.isolate, self.keep, self.sing = isolate, keep_workdir, singularity_bin
        self.max_container_s, self.canary, self.local_python = max_container_s, canary, local_python
        os.makedirs(run_dir, exist_ok=True)

    def _cmd(self, wd: str) -> list:
        if self.local_python:
            return [self.local_python, "ut_driver.py", "spec.json"]
        iso = ["--containall", "--no-home", "--cleanenv", "--net", "--network=none"] if self.isolate else []
        return [self.sing, "exec", *iso, "--writable-tmpfs", "--bind", f"{self.venv}:/ut_venv:ro", "--bind", f"{wd}:/work",
                "--pwd", "/work", "--env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1,PYTHONDONTWRITEBYTECODE=1,PYTHONHASHSEED=0,LANG=C.UTF-8",
                self.sif, "/ut_venv/bin/python", "ut_driver.py", "spec.json"]

    def run(self, inst: dict, code: str) -> ExecutionResult:
        t0 = time.time()
        tests = inst.get("tests") or []
        if not tests:
            return ExecutionResult(False, ErrKind.INFRA_DATA, "no_tests")
        if not code.strip():
            return ExecutionResult(False, ErrKind.FORMAT_NO_PATCH, "no_code")
        if not self.local_python and not os.path.exists(self.sif):
            return ExecutionResult(False, ErrKind.INFRA_ENV, f"sif missing: {self.sif}")
        wd = tempfile.mkdtemp(dir=self.run_dir, prefix="ut." + str(inst.get("instance_id", ""))[:40] + ".")
        try:
            shutil.copy(DRIVER, os.path.join(wd, "ut_driver.py"))
            spec = {"harness": inst.get("harness") or "function", "code": code, "tests": tests,
                    "time_limit_s": float(inst.get("time_limit_s") or 10), "memory_mb": int(inst.get("memory_mb") or 512),
                    "canary": self.canary}
            with open(os.path.join(wd, "spec.json"), "w", encoding="utf-8") as f:
                json.dump(spec, f, ensure_ascii=False)
            budget = min(self.max_container_s, int(spec["time_limit_s"] * max(3, len(tests)) + 60))
            try:
                r = subprocess.run(self._cmd(wd), cwd=wd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   timeout=budget)
                out = (r.stdout or "") + (r.stderr or "")
            except subprocess.TimeoutExpired:
                return ExecutionResult(False, ErrKind.INFRA_TIMEOUT, "container_timeout", timed_out=True, runtime_s=time.time() - t0)
            line = next((l for l in out.splitlines() if l.startswith("__UT_RESULT__")), None)
            if line is None:
                return ExecutionResult(False, ErrKind.INFRA_ENV, "driver_no_result", log_tail=out[-1500:], runtime_s=time.time() - t0)
            d = json.loads(line[len("__UT_RESULT__"):])
            if d.get("canary") == "cheat":
                return ExecutionResult(False, ErrKind.INVALID_ALWAYS_TRUE_EQ, "canary_cheat",
                                       f2p=TestOutcome((), tuple(d["passed"] + d["failed"]), ()), hack_flags=("always_true_eq",),
                                       runtime_s=time.time() - t0, log_tail=out[-1000:])
            if d.get("err") == "exec_error":
                return ExecutionResult(False, ErrKind.INFRA_OTHER, "exec_error", log_tail=out[-1500:], runtime_s=time.time() - t0)
            f2p = TestOutcome(tuple(d["passed"]), tuple(d["failed"]), ())
            return ExecutionResult(True, ErrKind.OK, "", f2p, TestOutcome(), runtime_s=time.time() - t0,
                                   timed_out=bool(d.get("timeouts")), log_tail=out[-800:])
        finally:
            if not self.keep:
                shutil.rmtree(wd, ignore_errors=True)
