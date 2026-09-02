"""Driver-level tests run with the host interpreter (no container) via local_python."""
import sys, os
import pytest
from verifier.ut_runner import UnitTestRunner
from verifier.schemas import ErrKind

PY = sys.executable
FUNC_INST = {"instance_id": "f1", "harness": "function", "time_limit_s": 5, "memory_mb": 512,
             "tests": [{"assertion": "assert add(1, 2) == 3"}, {"assertion": "assert add(0, 0) == 0"},
                       {"assertion": "assert add(-1, 1) == 0"}, {"assertion": "assert add(2, 2) == 5"}]}
PYTEST_INST = {"instance_id": "p1", "harness": "pytest", "time_limit_s": 5, "memory_mb": 512,
               "tests": [{"assertion": "from solution import add\n\ndef test_a():\n    assert add(1,2)==3\n\ndef test_b():\n    assert add(1,1)==3\n"}]}
STDIO_INST = {"instance_id": "s1", "harness": "stdio", "time_limit_s": 5, "memory_mb": 512,
              "tests": [{"stdin": "1 2\n", "stdout": "3"}, {"stdin": "5 5\n", "stdout": "10\n"}, {"stdin": "1 1\n", "stdout": "3"}]}


@pytest.fixture
def runner(tmp_path):
    return UnitTestRunner("/nonexistent.sif", "/nonexistent", str(tmp_path), local_python=PY)


def test_function_partial_credit(runner):
    r = runner.run(FUNC_INST, "def add(a, b):\n    return a + b\n")
    assert r.ran and r.f2p.n_passed == 3 and r.f2p.total == 4 and not r.resolved


def test_pytest_harness(runner):
    r = runner.run(PYTEST_INST, "def add(a, b):\n    return a + b\n")
    assert r.ran and set(r.f2p.passed) == {"test_0.py::test_a"} and set(r.f2p.failed) == {"test_0.py::test_b"}


def test_stdio_harness(runner):
    r = runner.run(STDIO_INST, "a, b = map(int, input().split())\nprint(a + b)\n")
    assert r.ran and r.f2p.n_passed == 2 and r.f2p.total == 3


def test_always_true_eq_canary_blocks(runner):
    cheat = ("class _Any:\n    def __eq__(self, o): return True\n    def __ne__(self, o): return False\n"
             "    def __getattr__(self, n): return _Any()\n    def __call__(self, *a, **k): return _Any()\n"
             "def add(*a, **k): return _Any()\n")
    r = runner.run(FUNC_INST, cheat)
    assert not r.ran and r.err_kind == ErrKind.INVALID_ALWAYS_TRUE_EQ
    r = runner.run(PYTEST_INST, cheat)
    assert not r.ran and r.err_kind == ErrKind.INVALID_ALWAYS_TRUE_EQ


def test_timeout_is_model_fault_not_infra(runner):
    r = runner.run({**FUNC_INST, "time_limit_s": 1}, "def add(a, b):\n    while True: pass\n")
    assert r.ran and r.f2p.n_passed == 0 and r.timed_out


def test_syntax_error_code_fails_all(runner):
    r = runner.run(FUNC_INST, "def add(a, b:\n    return a+b\n")
    assert r.ran and r.f2p.n_passed == 0


def test_no_code_and_no_tests(runner):
    assert runner.run(FUNC_INST, "").err_kind == ErrKind.FORMAT_NO_PATCH
    assert runner.run({**FUNC_INST, "tests": []}, "x=1").err_kind == ErrKind.INFRA_DATA


def test_type_conditional_and_partial_cheats_blocked(runner):
    cheat_str = ("class _Any:\n    def __eq__(self, o): return not isinstance(o, str)\n    def __ne__(self, o): return isinstance(o, str)\n"
                 "def add(*a, **k): return _Any()\n")
    r = runner.run(FUNC_INST, cheat_str)
    assert not r.ran and r.err_kind == ErrKind.INVALID_ALWAYS_TRUE_EQ
    partial = ("class _Any:\n    def __eq__(self, o): return True\n    def __ne__(self, o): return False\n"
               "def add(a, b):\n    return 3 if (a, b) == (1, 2) else _Any()\n")
    r = runner.run(FUNC_INST, partial)
    assert not r.ran and r.err_kind == ErrKind.INVALID_ALWAYS_TRUE_EQ
    r = runner.run(PYTEST_INST, cheat_str)
    assert not r.ran and r.err_kind == ErrKind.INVALID_ALWAYS_TRUE_EQ
    # honest custom classes with a normal __eq__ are not flagged
    honest = ("class V:\n    def __init__(self, x): self.x = x\n    def __eq__(self, o): return isinstance(o, V) and o.x == self.x\n"
              "def add(a, b): return a + b\n")
    r = runner.run(FUNC_INST, honest)
    assert r.ran and r.f2p.n_passed == 3
