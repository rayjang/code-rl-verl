"""pytest `-rA` summary parsing shared by the SWE and unit-test runners.

Traps inherited from the baseline (rl_code_v1/reward/swe_exec_v2.py, unittest_exec.py):
  * ANSI colour codes forced by repo config wrap the summary lines -> strip before matching.
  * A test id that is a parametrised prefix ("mod.py::test_x") passes only if EVERY parameter
    ("mod.py::test_x[...]") passed (conservative).
  * Zero parsed cases must be split into 'pytest never ran' (infra) vs 'collected nothing' (model).
"""
from __future__ import annotations

import re

from .schemas import TestOutcome

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
CASE_RE = re.compile(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\s+(\S+)")
BROKEN_RE = re.compile(r"No module named pytest|INTERNALERROR|unrecognized arguments|ImportError while loading conftest|usage: pytest")
RAN_RE = re.compile(r"collected \d+ item|no tests ran|=+ (test session starts|ERRORS|FAILURES)")


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s or "")


def clean_ids(ids) -> list:
    """Drop id fragments produced by whitespace splitting inside parameters, truncate cut
    parameters to function level, dedupe (port of swe_exec_v2._clean_ids)."""
    out = []
    for t in ids or []:
        if "::" not in t:
            continue
        if "[" in t and not t.endswith("]"):
            t = t.split("[", 1)[0]
        if t not in out:
            out.append(t)
    return out


def parse_summary(out: str) -> dict:
    """{test_id: STATUS} from `-rA` summary lines (ANSI stripped)."""
    res = {}
    for line in strip_ansi(out).splitlines():
        m = CASE_RE.match(line.strip())
        if m:
            res[m.group(2)] = m.group(1)
    return res


def pytest_ran(out: str) -> bool:
    c = strip_ansi(out)
    return bool(RAN_RE.search(c)) and not BROKEN_RE.search(c)


def evaluate(res: dict, test_ids, pass_statuses=("PASSED",)) -> TestOutcome:
    passed, failed, missing = [], [], []
    for t in test_ids:
        matches = [s for k, s in res.items() if k == t or k.startswith(t + "[") or k.startswith(t + "::")]
        if not matches:
            missing.append(t)
        elif all(s in pass_statuses for s in matches):
            passed.append(t)
        else:
            failed.append(t)
    return TestOutcome(tuple(passed), tuple(failed), tuple(missing))
