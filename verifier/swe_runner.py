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
import re
TESTISH_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")
from .schemas import ApplyResult, ApplyStatus, ErrKind, ExecutionResult, TestOutcome

SING = os.environ.get("SINGULARITY_BIN") or shutil.which("apptainer") or shutil.which("singularity") \
    or "/apps/applications/singularity/4.3.4/bin/singularity"


class SweSmithRunner:
    def __init__(self, sif_dir: str, run_dir: str, cache_dir: str, *, timeout: int = 900, extract_timeout: int = 300,
                 p2p_cap: int = 30, apply_mode: str = "strict", isolate: bool = True, keep_workdir: bool = False,
                 singularity_bin: str = SING, ignore_whitespace: bool = False, use_p2p_effective: bool = True):
        self.sif_dir, self.run_dir, self.cache_dir = sif_dir, run_dir, cache_dir
        self.timeout, self.extract_timeout, self.p2p_cap = timeout, extract_timeout, p2p_cap
        self.apply_mode, self.isolate, self.keep = apply_mode, isolate, keep_workdir
        self.ignore_whitespace, self.use_p2p_effective = ignore_whitespace, use_p2p_effective
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
        p2p = clean_ids(p2p)
        f2p = clean_ids(f2p)
        # ids inside a package __init__.py cannot be selected by path::id (pytest treats the file as a package
        # marker and errors the whole session, observed on joke2k__faker): drop them from both sets
        f2p = [t for t in f2p if not t.split("::")[0].endswith("__init__.py")]
        p2p = [t for t in p2p if not t.split("::")[0].endswith("__init__.py")]
        eff_f = inst.get("f2p_effective")        # ids that fail on the buggy tree AND pass with gold (validation run)
        if self.use_p2p_effective and isinstance(eff_f, list) and eff_f:
            s_f = set(eff_f)
            f2p = [t for t in f2p if t in s_f]
        eff = inst.get("p2p_effective")          # tests verified to pass on the unpatched tree (validation run)
        if self.use_p2p_effective and isinstance(eff, list) and eff:
            eff_set = set(eff)
            p2p = [t for t in p2p if t in eff_set]
        return f2p, p2p[: self.p2p_cap]

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
            # Restore ONLY test material: test-pattern files named by the ids, the test directories that
            # contain them, and the root conftest.py. Ids that point at source modules (doctest ids such as
            # `inflect/__init__.py::inflect.engine.compare`) are NOT restored -- restoring them would revert
            # the injected bug (observed on jaraco__inflect, 40/40 gold apply failures).
            testish = [t for t in test_files if TESTISH_RE.search(t)]
            test_dirs = sorted({t.split("/")[0] for t in testish if "/" in t and t.split("/")[0] in ("tests", "test", "testing")})
            tf_args = " ".join(shlex.quote(t) for t in testish)
            restore_args = " ".join(shlex.quote(d) for d in test_dirs + ["conftest.py"])
            # older git inside the images ignores `-c safe.directory`; a global config file is honoured
            with open(f"{wd}/gitconfig", "w") as f:
                f.write("[safe]\n\tdirectory = *\n")
            shutil.copy(f"{wd}/gitconfig", f"{wd}/.gitconfig")
            # Copy the image's /testbed (keeps untracked build artifacts such as generated _version.py
            # and compiled extensions that `git archive` would drop), switch the tracked files to the
            # instance branch (buggy state), restore the deleted test files from main, drop .git.
            # tests are deleted on the instance branch; restore the whole test tree from main (helper modules,
            # package __init__ files and conftest.py included) -- restoring only the named files leaves
            # collection errors ("33 errors in 0.06s") for repos whose tests import sibling helpers.
            script = (f"cp -a /testbed /wd/tb && cd /wd/tb && git checkout -q -f origin/{shlex.quote(iid)} && "
                      f"(git checkout -q main -- {restore_args} 2>/dev/null || true) && "
                      + (f"git checkout -q main -- {tf_args} && " if tf_args else "")
                      + "rm -rf /wd/tb/.git && echo EXTRACT_OK")
            try:
                r = subprocess.run([self.sing, "exec", *self._iso_flags(), "--bind", f"{wd}:/wd",
                                    "--env", "HOME=/wd,GIT_CONFIG_GLOBAL=/wd/gitconfig", sif, "bash", "-c", script],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=self.extract_timeout)
            except subprocess.TimeoutExpired:
                shutil.rmtree(wd, ignore_errors=True)
                return None, "extract_timeout"
            if r.returncode != 0 or "EXTRACT_OK" not in r.stdout:
                err = (r.stderr or r.stdout)[-400:]
                shutil.rmtree(wd, ignore_errors=True)
                return None, f"branch_extract_fail: {err}"
            missing = [t for t in testish if not os.path.exists(os.path.join(wd, "tb", t))]
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
                ap = apply_patch(patch, tree, mode=self.apply_mode, reverse=reverse, ignore_whitespace=self.ignore_whitespace)
            else:
                ap = ApplyResult(ApplyStatus.PATCH_APPLY_SUCCESS, (), "empty patch (no-op)")
            if ap.status == ApplyStatus.PATCH_APPLY_FAIL:
                return ExecutionResult(False, ErrKind.FORMAT_APPLY_FAIL, "patch_apply_fail", apply=ap,
                                       log_tail=ap.stderr_tail, runtime_s=time.time() - t0)
            tests = " ".join(shlex.quote(t) for t in (f2p + p2p))
            doctest_flag = " --doctest-modules" if any(not TESTISH_RE.search(t.split("::")[0]) for t in (f2p + p2p)) else ""
            env_lines = "".join(f"export {k}={shlex.quote(str(v))}\n" for k, v in (extra_env or {}).items())
            runner = ("#!/bin/bash\n"
                      "source /opt/miniconda3/bin/activate testbed 2>/dev/null || export PATH=/opt/miniconda3/envs/testbed/bin:$PATH\n"
                      f"{env_lines}"
                      # NOTE: never export PYTEST_DISABLE_PLUGIN_AUTOLOAD here -- pytest disables autoload on ANY non-empty
                      # value (even "0"); repos such as Red-DiscordBot (pytest-asyncio) and typeguard (own plugin) need plugins.
                      "export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0; unset PYTEST_DISABLE_PLUGIN_AUTOLOAD\n"
                      "cd /testbed\n"
                      f"timeout -s KILL {max(30, self.timeout - 15)} python -m pytest -rA --tb=no --color=no -p no:cacheprovider{doctest_flag} -q {tests} 2>&1\n"
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
            crashed = ()
            if not res and not pytest_ran(out):
                # pytest could not even start. The environment itself is validated (gold runs), so on the
                # patched tree this is the patch's fault (e.g. the package under test is imported by pytest,
                # as in agronholm__exceptiongroup): every F2P/P2P id counts as failed, flagged `pytest_crashed`.
                # Genuine environment breakage (image/pytest missing) is still surfaced as infra.
                low = out.lower()
            if not res and not pytest_ran(out) and ("no module named pytest" in low or "singularity" in low[:400] or "fatal:" in low[:400]):
                return ExecutionResult(False, ErrKind.INFRA_ENV, "pytest_did_not_run", apply=ap,
                                       log_tail=out[-2000:], runtime_s=time.time() - t0)
            if not res and not pytest_ran(out):
                crashed = ("pytest_crashed",)
            f2p_o = evaluate(res, f2p)
            p2p_o = evaluate(res, p2p)
            return ExecutionResult(True, ErrKind.OK, "", f2p_o, p2p_o, runtime_s=time.time() - t0,
                                   timed_out=("PYTEST_EXIT=137" in out), log_tail=out[-2500:], apply=ap, hack_flags=crashed)
        finally:
            if not self.keep:
                shutil.rmtree(wd, ignore_errors=True)
