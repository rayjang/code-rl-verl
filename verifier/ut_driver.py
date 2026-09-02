#!/usr/bin/env python3
"""In-container driver for the unit-test track. Self-contained (stdlib only); copied into the
work dir and executed with the sandbox interpreter:  python ut_driver.py spec.json

spec = {"harness": "function|pytest|stdio", "code": str, "tests": [...], "time_limit_s": float,
        "memory_mb": int, "canary": bool}
Prints one JSON line: {"passed": [...], "failed": [...], "err": "", "canary": "ok|cheat|skip",
                       "spawn_fail": n, "pytest_ran": bool}

Port of rl_code_v1/reward/unittest_exec.py semantics (per-assertion subprocesses for partial
credit, pytest -rA summary parsing with ANSI stripping, stdio normalised comparison, rlimits,
always-true __eq__ canary). Runs INSIDE the sandbox; isolation comes from the container flags.
"""
import ast, json, os, re, resource, subprocess, sys, textwrap

import secrets as _secrets
# randomised per run so that a cheating __eq__ cannot special-case the probe values; several types so that
# type-conditional cheats (e.g. `return not isinstance(o, str)`) are caught as well
SENTINEL = f"__CANARY_{_secrets.token_hex(8)}__"
SENTINELS = [SENTINEL, 7_000_000_000 + _secrets.randbelow(1_000_000_000), 1e9 + _secrets.randbelow(10**6) + 0.5,
             ("__canary__", _secrets.randbelow(10**9)), object()]
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
CASE = re.compile(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\s+(\S+)", re.M)
BROKEN = re.compile(r"No module named pytest|INTERNALERROR|unrecognized arguments|ImportError while loading conftest|usage: pytest")
RAN = re.compile(r"collected \d+ item|no tests ran|=+ (test session starts|ERRORS|FAILURES)")
PY = sys.executable


def limits(mem_mb):
    def _apply():
        b = mem_mb * 1024 * 1024
        for r in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            try:
                resource.setrlimit(r, (b, b))
            except (ValueError, OSError):
                pass
        for r, v in ((resource.RLIMIT_NPROC, 128), (resource.RLIMIT_FSIZE, 64 << 20), (resource.RLIMIT_CORE, 0)):
            try:
                resource.setrlimit(r, (v, v))
            except (ValueError, OSError):
                pass
        os.setsid()
    return _apply


def run(args, cwd, timeout, mem_mb, stdin=None):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": cwd, "TMPDIR": cwd, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONIOENCODING": "utf-8"}
    try:
        p = subprocess.run(args, cwd=cwd, input=stdin, capture_output=True, text=True, timeout=timeout,
                           preexec_fn=limits(mem_mb), env=env, encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return -9, "__TIMEOUT__"
    except Exception as e:
        return -1, f"__EXEC_ERROR__ {e}"


def norm_out(s):
    return "\n".join(line.rstrip() for line in (s or "").strip().split("\n"))


def canary_operands(assertion):
    """(setup_code, expr) for the first == / != compare inside the assertion; setup = statements
    before the assert (e.g. `acc = Bank(); acc.deposit(1); assert acc.balance() == 1`)."""
    try:
        tree = ast.parse(assertion)
    except SyntaxError:
        return None
    body = list(tree.body)
    for idx, stmt in enumerate(body):
        if isinstance(stmt, ast.Assert):
            t = stmt.test
            if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], (ast.Eq, ast.NotEq)):
                expr = t.left if not isinstance(t.left, ast.Constant) else t.comparators[0]
                if isinstance(expr, ast.Constant):
                    return None
                setup = ast.unparse(ast.Module(body=body[:idx], type_ignores=[])) if idx else ""
                return setup, ast.unparse(expr)
            if isinstance(t, ast.Call):
                setup = ast.unparse(ast.Module(body=body[:idx], type_ignores=[])) if idx else ""
                return setup, ast.unparse(t)
    return None


def canary_lhs(assertion):
    try:
        tree = ast.parse(assertion)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            try:
                return ast.unparse(node.left)
            except Exception:
                return None
    return None


CANARY_FN = """
import sys
try:
    exec(open({sol!r}, encoding='utf-8').read(), globals())
except Exception:
    print("CANARY_SKIP"); sys.exit(0)
try:
    exec({setup!r}, globals())
    _v = {lhs}
except Exception:
    print("CANARY_SKIP"); sys.exit(0)
_bad = False
if not isinstance(_v, (bool, int, float, str, bytes, tuple, list, dict, set, frozenset, type(None))):
    for _s in [{sent!r}, {sent_int!r}, {sent_float!r}, {sent_tuple!r}, object()]:
        try:
            if bool(_v == _s) or bool(_s == _v) or bool(_v != _s) is False:
                _bad = True; break
        except Exception:
            pass
print("CANARY_CHEAT" if _bad else "CANARY_OK")
"""
CONFTEST = """
import builtins
S = {sent!r}
class CanaryCheat(Exception):
    pass
SENTS = [S, {sent_int!r}, {sent_float!r}, {sent_tuple!r}, object()]
def _probe(a):
    if isinstance(a, (bool, int, float, str, bytes, tuple, list, dict, set, frozenset, type(None))):
        return                      # builtin values cannot lie
    for s_ in SENTS:
        try:
            bad = bool(a == s_) or bool(s_ == a) or (bool(a != s_) is False)
        except CanaryCheat:
            raise
        except Exception:
            bad = False
        if bad:
            raise CanaryCheat("CANARY_CHEAT always-true comparison")
def __t15_eq(a, b):
    _probe(a); _probe(b)
    return a == b
def __t15_ne(a, b):
    _probe(a); _probe(b)
    return a != b
def __t15_truth(a):
    _probe(a)
    return a
builtins.__t15_eq = __t15_eq
builtins.__t15_ne = __t15_ne
builtins.__t15_truth = __t15_truth
"""


class _AssertRewriter(ast.NodeTransformer):
    """assert a == b  ->  assert __t15_eq(a, b);  assert a != b -> __t15_ne;  assert f(x) -> __t15_truth(f(x))."""
    def visit_Assert(self, node):
        self.generic_visit(node)
        t = node.test
        if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], (ast.Eq, ast.NotEq)):
            fn = "__t15_eq" if isinstance(t.ops[0], ast.Eq) else "__t15_ne"
            node.test = ast.Call(ast.Name(fn, ast.Load()), [t.left, t.comparators[0]], [])
        elif isinstance(t, ast.Call):
            node.test = ast.Call(ast.Name("__t15_truth", ast.Load()), [t], [])
        return ast.fix_missing_locations(node)


def rewrite_test_source(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    return ast.unparse(_AssertRewriter().visit(tree))


CANARY_PYTEST = """
import sys
try:
    import solution
except Exception:
    print("CANARY_SKIP"); sys.exit(0)
S = {sent!r}
bad = False
for _n in dir(solution):
    try:
        o = getattr(solution, _n)
        if not isinstance(o, (type, bool, int, float, str, bytes, tuple, list, dict, set, frozenset, type(None))) and bool(o == S):
            bad = True; break
    except Exception:
        pass
if not bad:
    try:
        o = getattr(solution, "__t15_canary_probe__", None)
        if o is not None and bool(o == S):
            bad = True
    except Exception:
        pass
print("CANARY_CHEAT" if bad else "CANARY_OK")
"""


def run_canary(wd, harness, tests, timeout, mem_mb):
    if harness == "pytest":
        src = CANARY_PYTEST.format(sent=SENTINEL)
        with open(f"{wd}/_canary.py", "w", encoding="utf-8") as f:
            f.write(src)
        _, out = run([PY, "_canary.py"], wd, timeout, mem_mb)
        if "CANARY_CHEAT" in out:
            return "cheat"
        return "ok" if "CANARY_OK" in out else "skip"
    # function harness: probe the operands of every == / != assertion (cheap: one process each)
    probes = [x for x in (canary_operands(t.get("assertion") or "") for t in tests) if x]
    if not probes:
        return "skip"
    verdict = "skip"
    for setup, expr in probes[:16]:
        src = CANARY_FN.format(sol="_sol.py", setup=setup, lhs=expr, sent=SENTINEL, sent_int=SENTINELS[1], sent_float=SENTINELS[2], sent_tuple=SENTINELS[3])
        with open(f"{wd}/_canary.py", "w", encoding="utf-8") as f:
            f.write(src)
        _, out = run([PY, "_canary.py"], wd, timeout, mem_mb)
        if "CANARY_CHEAT" in out:
            return "cheat"
        if "CANARY_OK" in out:
            verdict = "ok"
    return verdict


def main():
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    wd = os.getcwd()
    harness, code, tests = spec.get("harness") or "function", spec["code"], spec.get("tests") or []
    timeout, mem_mb = float(spec.get("time_limit_s") or 10), int(spec.get("memory_mb") or 512)
    res = {"passed": [], "failed": [], "err": "", "canary": "skip", "spawn_fail": 0, "pytest_ran": True, "timeouts": 0}
    if harness == "pytest":
        open(f"{wd}/solution.py", "w", encoding="utf-8").write(code)
        expected = []
        for i, t in enumerate(tests):
            src = t.get("assertion") or ""
            open(f"{wd}/test_{i}.py", "w", encoding="utf-8").write(rewrite_test_source(src) if spec.get("canary", True) else src)
            for m in re.finditer(r"^\s*def\s+(test\w*)\s*\(", src, re.M):
                expected.append(f"test_{i}.py::{m.group(1)}")
        if spec.get("canary", True):
            open(f"{wd}/conftest.py", "w", encoding="utf-8").write(CONFTEST.format(sent=SENTINEL, sent_int=SENTINELS[1], sent_float=SENTINELS[2], sent_tuple=SENTINELS[3]))
        rc, out = run([PY, "-m", "pytest", "-rA", "--tb=line", "--color=no", "-p", "no:cacheprovider", "-q", "."], wd,
                      max(timeout * 3, 30), mem_mb)
        if out == "__TIMEOUT__":
            res["err"] = "timeout"; res["failed"] = expected; res["timeouts"] = 1
        elif out.startswith("__EXEC_ERROR__"):
            res["err"] = "exec_error"; res["pytest_ran"] = False
        else:
            clean = ANSI.sub("", out)
            found = {tid: st for st, tid in CASE.findall(clean)}
            if not found and (BROKEN.search(clean) or not RAN.search(clean)):
                res["err"] = "exec_error"; res["pytest_ran"] = False
            else:
                # parametrised ids ("test_0.py::test_a[1]") map back to their function id
                for tid in expected or list(found):
                    ms = [st for k, st in found.items() if k == tid or k.endswith("::" + tid.split("::")[-1]) or k.startswith(tid + "[")]
                    (res["passed"] if ms and all(s in ("PASSED", "XFAIL") for s in ms) else res["failed"]).append(tid)
                if spec.get("canary", True):
                    if "CanaryCheat" in clean or "CANARY_CHEAT" in clean:
                        res["canary"] = "cheat"
                    elif res["passed"]:
                        res["canary"] = run_canary(wd, "pytest", tests, timeout, mem_mb)
    elif harness == "stdio":
        open(f"{wd}/main.py", "w", encoding="utf-8").write(code)
        for i, t in enumerate(tests):
            rc, out = run([PY, "main.py"], wd, timeout, mem_mb, stdin=(t.get("stdin") or ""))
            tid = f"case_{i}"
            if out.startswith("__EXEC_ERROR__"):
                res["spawn_fail"] += 1; res["failed"].append(tid); continue
            if out == "__TIMEOUT__":
                res["timeouts"] += 1; res["failed"].append(tid); continue
            (res["passed"] if rc == 0 and norm_out(out) == norm_out(t.get("stdout") or "") else res["failed"]).append(tid)
    else:  # function
        open(f"{wd}/_sol.py", "w", encoding="utf-8").write(code)
        for i, t in enumerate(tests):
            a = t.get("assertion") or ""
            runner = textwrap.dedent(f"""
                import sys
                sys.setrecursionlimit(20000)
                exec(open('_sol.py', encoding='utf-8').read(), globals())
                {a}
            """)
            open(f"{wd}/_t.py", "w", encoding="utf-8").write(runner)
            rc, out = run([PY, "_t.py"], wd, timeout, mem_mb)
            tid = f"case_{i}"
            if out.startswith("__EXEC_ERROR__"):
                res["spawn_fail"] += 1; res["failed"].append(tid); continue
            if out == "__TIMEOUT__":
                res["timeouts"] += 1; res["failed"].append(tid); continue
            (res["passed"] if rc == 0 else res["failed"]).append(tid)
        if spec.get("canary", True) and res["passed"]:
            res["canary"] = run_canary(wd, "function", tests, timeout, mem_mb)
    total = len(res["passed"]) + len(res["failed"])
    if res["spawn_fail"] and res["spawn_fail"] >= max(1, total // 2):
        res["err"] = "exec_error"
    print("__UT_RESULT__" + json.dumps(res))


if __name__ == "__main__":
    main()
