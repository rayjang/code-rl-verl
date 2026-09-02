"""Host-side patch application against an extracted (non-git) source tree.

`git apply` works outside a repository as a strict patch tool: no fuzz, exact context match,
rejects corrupt hunks. Two modes:
  strict   (default)  `git apply --check` then `git apply`. Never fuzzes.
  fuzz     (ablation) `git apply` then `patch -p1 --fuzz=5` fallback, i.e. the baseline behaviour.
                      Applying with fuzz silently changes where hunks land -> reported as
                      `apply_mode=fuzz` so the ablation can measure how often it 'rescues' patches.
Statuses recorded: git_apply_check_success / patch_apply_success / patch_apply_fail.
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional

from .diff_parser import touched_paths
from .schemas import ApplyResult, ApplyStatus

GIT_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "LC_ALL": "C.UTF-8", "HOME": "/nonexistent"}


def _run(cmd, cwd, timeout=60, input_bytes: Optional[bytes] = None):
    return subprocess.run(cmd, cwd=cwd, input=input_bytes, capture_output=True, timeout=timeout,
                          env={**os.environ, **GIT_ENV})


def git_apply_check(patch: str, tree_dir: str, timeout: int = 60) -> ApplyResult:
    files = touched_paths(patch)
    try:
        r = _run(["git", "apply", "--check", "--unsafe-paths", "-p1", "-"], tree_dir, timeout, patch.encode("utf-8"))
    except subprocess.TimeoutExpired:
        return ApplyResult(ApplyStatus.PATCH_APPLY_FAIL, files, "git apply --check timeout")
    if r.returncode == 0:
        return ApplyResult(ApplyStatus.GIT_APPLY_CHECK_SUCCESS, files)
    return ApplyResult(ApplyStatus.PATCH_APPLY_FAIL, files, "git apply --check failed",
                       r.stderr.decode("utf-8", "replace")[-800:])


def apply_patch(patch: str, tree_dir: str, mode: str = "strict", reverse: bool = False, timeout: int = 120) -> ApplyResult:
    """Apply `patch` in-place under tree_dir. Returns ApplyResult with final status."""
    files = touched_paths(patch)
    if not patch.strip():
        return ApplyResult(ApplyStatus.PATCH_APPLY_FAIL, files, "empty patch")
    rflag = ["-R"] if reverse else []
    try:
        chk = _run(["git", "apply", "--check", "--unsafe-paths", "-p1", *rflag, "-"], tree_dir, timeout, patch.encode())
        if chk.returncode == 0:
            r = _run(["git", "apply", "--unsafe-paths", "-p1", *rflag, "-"], tree_dir, timeout, patch.encode())
            if r.returncode == 0:
                return ApplyResult(ApplyStatus.PATCH_APPLY_SUCCESS, files)
            return ApplyResult(ApplyStatus.PATCH_APPLY_FAIL, files, "git apply failed after successful check",
                               r.stderr.decode("utf-8", "replace")[-800:])
        if mode != "fuzz":
            return ApplyResult(ApplyStatus.PATCH_APPLY_FAIL, files, "git apply --check failed",
                               chk.stderr.decode("utf-8", "replace")[-800:])
        # ---- baseline fallback (ablation only): GNU patch with fuzz ----
        r = _run(["patch", "-p1", "--fuzz=5", "--no-backup-if-mismatch", "-f", *(["-R"] if reverse else []), "-i", "-"],
                 tree_dir, timeout, patch.encode())
        if r.returncode == 0:
            res = ApplyResult(ApplyStatus.PATCH_APPLY_SUCCESS, files, "applied with patch --fuzz=5 (repair path)")
            return res
        return ApplyResult(ApplyStatus.PATCH_APPLY_FAIL, files, "git apply and patch --fuzz both failed",
                           (chk.stderr + r.stdout + r.stderr).decode("utf-8", "replace")[-800:])
    except subprocess.TimeoutExpired:
        return ApplyResult(ApplyStatus.PATCH_APPLY_FAIL, files, "apply timeout")
    except FileNotFoundError as e:
        return ApplyResult(ApplyStatus.PATCH_APPLY_FAIL, files, f"tool missing: {e}")
