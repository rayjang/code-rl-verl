import sys, os, json
sys.path.insert(0, "/scratch/r919a03/code_verl_test/sources/rl_code_v1/reward")
os.environ.setdefault("RL_INDEX_DIR", "/nonexistent")
import t15_code_judge as CJ
import patch_extract as PE
import unittest_exec as UT
import swe_exec_v2 as V2
import t15_code_reward as R

print("== score_of ladder")
cases = [("no_patch",0,1),("apply_fail",0,1),("invalid",0,1),("ran",0.0,1.0),("ran",0.2,1.0),("ran",0.5,1.0),("ran",0.8,1.0),("ran",0.9,1.0),
         ("ran",1.0,1.0),("ran",1.0,29/30),("ran",1.0,27/30),("ran",1.0,15/30),("ran",1.0,0.0),("ran",0.5,0.0),("ran",0.0,0.0),("ran",0.1,1.0),("ran",1.0,1.5),("ran",-0.5,1.0),("ran",0.5,29/30),("ran",1.0,0.5), ("ran", 1.0, 0.99999)]
for c in cases:
    print(f"  score_of{c} = {CJ.score_of(*c):.6f}")
# P2P_MAX 0.60 counterfactual
reg=1.0; print("  counterfactual P2P_MAX=0.60, f2p all, reg 1:", 0.85*(1-0.60*reg**1.5))
print("  where floor kicks in (f2p=0, p2p_frac): ", [(p, round(CJ.score_of('ran',0.0,p),4)) for p in (1.0,0.5,0.2,0.1,0.0)])
print("  1/30 mult:", 1-0.75*((1/30)**1.5), "15/30 mult:", 1-0.75*(0.5**1.5))

print("== _err_kind")
for e in ["", "empty_patch","no_patch","patch_apply_fail","patch_touches_tests","timeout","extract_timeout","sif 없음: x","branch_extract_fail: y","unknown_instance","canary_cheat","no_valid_f2p","local_repo 없음: z","test_restore_fail: q","sif exec 불가: e","weird"]:
    print(f"  {e!r:32} -> {CJ._err_kind(e)}")

print("== _blank")
print(" ", CJ._blank("empty_patch","none"))
print(" ", CJ._blank("patch_apply_fail","fenced_diff","apply_fail"))
print(" ", CJ._blank("patch_touches_tests","raw_git","invalid"))

print("== touched_paths / touches_tests")
p1 = "diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n-a\n+b\n"
p2 = "diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n"
p3 = "diff --git q/tests/test_x.py r/tests/test_x.py\n--- q/tests/test_x.py\n+++ r/tests/test_x.py\n@@ -1 +1 @@\n-a\n+b\n"
p4 = "--- /dev/null\n+++ b/newfile_test.py\n"
p5 = "diff --git a/Tests/foo.py b/Tests/foo.py\n"
p6 = "diff --git a/src/pyproject.toml b/src/pyproject.toml\n"
p7 = "diff --git a/src/mod.py b/src/mod.py\n"
for lab,p in [("src",p1),("tests/",p2),("q/ r/ prefix",p3),("/dev/null new _test.py",p4),("Tests/ capital",p5),("nested pyproject",p6),("f2p file match",p7)]:
    print(f"  {lab:24} paths={CJ.touched_paths(p)} touches={CJ.touches_tests(p, ['src/mod.py::test_a'] if lab=='f2p file match' else [])}")
for path in ["tests/a.py","test/a.py","testing/a.py","a/test_x.py","a/x_test.py","conftest.py","a/conftest.py","pytest.cfg","tox.cfg","setup.cfg","pyproject.toml","a/pyproject.toml","src/contest.py","src/latest/x.py","attest.py","mytests/x.py","setup.py","tests.py","test.py"]:
    print(f"  TESTISH {path:22} -> {bool(CJ._TESTISH.search(path))}")

print("== extract_patch cascade")
def ep(lab, s, bf=None):
    p,f = PE.extract_patch(s, bf); print(f"  {lab:28} fmt={f:14} patch={p!r}")
ep("empty", "")
ep("prose only", "I think the fix is trivial.")
ep("fenced diff", "text\n```diff\ndiff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n```\nafter")
ep("fenced patch", "```patch\n--- a/x\n+++ b/x\n```")
ep("fenced python w/ diff", "```python\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n```")
ep("fenced python no diff", "```python\nprint(1)\n```")
ep("two fenced diffs -> last", "```diff\n--- a/one\n+++ b/one\n```\n```diff\n--- a/two\n+++ b/two\n```")
ep("fenced_diff beats later raw", "```diff\n--- a/one\n+++ b/one\n```\ndiff --git a/two b/two\n--- a/two\n+++ b/two\n")
ep("raw git", "explain\ndiff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\nthanks")
ep("two raw git -> last", "diff --git a/one b/one\n--- a/one\n+++ b/one\ndiff --git a/two b/two\n--- a/two\n+++ b/two\n")
ep("raw unified", "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n")
ep("raw unified /dev/null", "--- /dev/null\n+++ b/x\n@@ -0,0 +1 @@\n+b\n")
ep("CRLF fenced diff", "```diff\r\n--- a/x\r\n+++ b/x\r\n```\r\n")
ep("fence with CR only", "```diff\r--- a/x\r+++ b/x\r```")
ep("begin patch", "*** Begin Patch\n*** Update File: src/x.py\n@@ def f\n-a\n+b\n*** End Patch")
ep("begin patch add file", "*** Begin Patch\n*** Add File: src/new.py\n+print(1)\n*** End Patch")
ep("begin patch no file", "*** Begin Patch\nnothing\n*** End Patch")
bf = {"src/x.py": "def f():\n    return 1\n\n\ndef g():\n    return 2\n", "pkg/util.py": "x = 1\n"}
sr = "<solution>\n### src/x.py\n<<<<<<< SEARCH\n    return 1\n=======\n    return 10\n>>>>>>> REPLACE\n</solution>"
ep("search/replace exact", sr, bf)
ep("search/replace no buggy_files", sr, None)
ep("search/replace basename fuzzy", "### x.py\n<<<<<<< SEARCH\n    return 1\n=======\n    return 10\n>>>>>>> REPLACE", bf)
ep("search/replace ambiguous", "### src/x.py\n<<<<<<< SEARCH\n    return\n=======\n    return 10\n>>>>>>> REPLACE", bf)
ep("s/r one bad one good", "### src/x.py\n<<<<<<< SEARCH\nNOT PRESENT\n=======\nzzz\n>>>>>>> REPLACE\n### src/x.py\n<<<<<<< SEARCH\n    return 2\n=======\n    return 20\n>>>>>>> REPLACE", bf)
ep("s/r unknown path", "### nope.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE", bf)
ep("s/r no-op", "### src/x.py\n<<<<<<< SEARCH\n    return 1\n=======\n    return 1\n>>>>>>> REPLACE", bf)
print("  _norm cases:", repr(PE._norm("\n\nab\r\ncd\n\n")), repr(PE._norm("")), repr(PE._norm("\n")))
print("  _from_begin_patch:", repr(PE._from_begin_patch("*** Update File: a.py\n@@ hunk\n ctx\n-a\n+b\n*** End of File")))

print("== unittest extract_code")
for lab,s in [("py fence","x\n```python\ndef f(): pass\n```"),("bare fence","```\nimport os\n```"),("two fences -> last","```python\ndef a(): pass\n```\n```py\ndef b(): pass\n```"),("diff fence","```diff\n--- a\n```"),("no fence but def","hello\ndef f():\n    return 1"),("prose","hello world"),("empty",""),("fence lang other","```javascript\ndef x(): pass\n```")]:
    print(f"  {lab:22} -> {UT.extract_code(s)!r}")
print("== _canary_lhs")
for a in ["assert f(1) == 2", "assert 2 == f(1)", "assert f(1) != 2", "assert f(1) == 2 == g()", "assert f(1) < 3", "assert (f(1)) == 2 and g() == 3", "assert abs(f(1) - 2) < 1e-9", "not python (", "assert f(1)==g(2)"]:
    print(f"  {a!r:36} -> {UT._canary_lhs(a)!r}")
print("== _norm_out")
print(" ", repr(UT._norm_out("a  \nb\n\n")), repr(UT._norm_out("  a\nb")), UT._norm_out("1 \n2\n")==UT._norm_out("1\n2"), UT._norm_out("a\r\nb")==UT._norm_out("a\nb"))
print("== _infra_guard")
for args in [(0,10,5),(0,10,4),(3,3,1),(0,1,1),(0,0,0),(2,4,2),(0,10,10)]:
    print(f"  {args} -> {UT._infra_guard(*args)}")
print("== pytest regexes")
sample = "\x1b[32mPASSED\x1b[0m tests/t.py::\x1b[1mtest_a\x1b[0m\nFAILED tests/t.py::test_b - AssertionError\nXFAIL tests/t.py::test_c\nSKIPPED tests/t.py::test_d\nPASSED tests/t.py::test_p[1]\nFAILED tests/t.py::test_p[2]\nPASSED tests/t.py::test_q[x]\nERROR tests/t.py::test_e\n"
clean = UT._ANSI.sub("", sample)
res = UT._PYTEST_CASE.findall(clean); print("  UT cases:", res, "npass(PASSED|XFAIL)=", sum(1 for s,_ in res if s in ("PASSED","XFAIL")))
print("  V2 parse:", V2._pytest_parse(sample, ["tests/t.py::test_a","tests/t.py::test_b","tests/t.py::test_c","tests/t.py::test_p","tests/t.py::test_q","tests/t.py::test_zz","tests/t.py"]))
print("  BROKEN:", [bool(UT._PYTEST_BROKEN.search(s)) for s in ["No module named pytest","INTERNALERROR> x","usage: pytest [options]","ok"]])
print("  RAN:", [bool(UT._PYTEST_RAN.search(s)) for s in ["collected 3 items","no tests ran in 0.1s","=== test session starts ===","== ERRORS ==","== FAILURES ==","nothing"]])
print("  def-count:", len(__import__('re').findall(r"^\s*def\s+test\w*\s*\(", "def test_a():\n  pass\ndef helper():\n  pass\n  def test_b ():\nclass T:\n    def test_c(self):", __import__('re').M)))
print("== _clean_ids")
print(" ", V2._clean_ids(["a.py::t1","fragment","a.py::t2[('xlrd',","a.py::t1","a.py::t3[ok]","a.py::t2[('xlrd',"]))
print("== _overlong (buffer=0 default)")
print(" ", R._overlong(0.5, {"valid_response_length": 100, "max_response_length": 200}))
print(" ", R._overlong(0.5, {"valid_response_length": 200, "max_response_length": 200}))
print(" ", R._overlong(0.5, {}))
print(" ", R._overlong(0.5, {"valid_response_length": None}))
print("== _clean")
print(" ", [R._clean(v) for v in [True, False, 3, 2.5, float('nan'), float('inf'), "s", None, [1], {"a":1}]])
print("== _KEYS count", len(R._KEYS), list(R._KEYS))
print("== compute_score_dict on SWE unknown instance (index missing)")
d = R.compute_score_dict("t15_repo_patch", "```diff\n--- a/x\n+++ b/x\n```", "nope", {"instance_id":"nope","valid_response_length":10,"max_response_length":100})
print("  ", d)
try:
    R.compute_score_dict("t15_unittest_impl", "x", "nope", {})
except Exception as e:
    print("  unittest with missing index raises:", type(e).__name__, str(e)[:80])
print("== judge_unittest with in-memory instance (function harness)")
inst = {"instance_id":"i1","harness":"function","tests":[{"assertion":"assert f(1) == 2"},{"assertion":"assert f(2) == 3"},{"assertion":"assert f(3) == 99"}],"time_limit_s":5}
print("  good:", UT.judge_unittest("i1","```python\ndef f(x): return x+1\n```",inst))
print("  cheat:", UT.judge_unittest("i1","```python\nclass _Any:\n    def __eq__(self,o): return True\n    def __getattr__(self,n): return _Any()\n    def __call__(self,*a,**k): return _Any()\ndef __getattr__(name): return _Any()\ndef f(x): return _Any()\n```",inst))
print("  nocode:", UT.judge_unittest("i1","no code here",inst))
print("  print __:", UT.judge_unittest("i1","```python\nprint('__hello')\ndef f(x): return x+1\n```",inst))
print("  zero tests:", UT.judge_unittest("i1","```python\ndef f(x): return x+1\n```",{"instance_id":"i","harness":"function","tests":[]}))
print("  stdio:", UT.judge_unittest("s","```python\nprint(input().strip()*2)\n```",{"harness":"stdio","tests":[{"stdin":"ab\n","stdout":"abab\n"},{"stdin":"x\n","stdout":"WRONG"}],"time_limit_s":5}))
print("  pytest:", UT.judge_unittest("p","```python\ndef f(x): return x+1\n```",{"harness":"pytest","tests":[{"assertion":"from solution import f\ndef test_a():\n    assert f(1)==2\ndef test_b():\n    assert f(1)==3\n"}],"time_limit_s":5}))
print("  pytest cheat:", UT.judge_unittest("p","```python\nclass _Any:\n    def __eq__(self,o): return True\n    def __getattr__(self,n): return _Any()\n    def __call__(self,*a,**k): return _Any()\ndef __getattr__(name): return _Any()\n```",{"harness":"pytest","tests":[{"assertion":"from solution import f\ndef test_a():\n    assert f(1)==2\n"}],"time_limit_s":5}))
print("  pytest hang:", UT.judge_unittest("p","```python\nwhile True: pass\n```",{"harness":"pytest","tests":[{"assertion":"from solution import f\ndef test_a():\n    assert f(1)==2\n"}],"time_limit_s":1}))
print("  function hang:", UT.judge_unittest("p","```python\ndef f(x):\n    while True: pass\n```",{"harness":"function","tests":[{"assertion":"assert f(1)==2"}],"time_limit_s":1}))
