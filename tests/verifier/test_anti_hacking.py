from verifier.anti_hacking import inspect_patch, classify_paths

INST = {"FAIL_TO_PASS": ["tests/test_a.py::test_x"], "PASS_TO_PASS": ["tests/test_b.py::test_y"]}


def d(path, added="+x = 1"):
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1,1 +1,2 @@\n a\n{added}\n"


def test_source_edit_ok():
    assert inspect_patch(d("pkg/mod.py"), INST) == ((), "ok")


def test_test_file_edit_invalid_unless_allowed():
    assert inspect_patch(d("tests/test_a.py"), INST)[1] == "invalid"
    assert inspect_patch(d("pkg/test_utils.py"), INST)[1] == "invalid"
    assert inspect_patch(d("tests/test_a.py"), {**INST, "allow_test_edits": True})[1] == "ok"


def test_config_and_conftest_and_escape():
    assert "edits_config" in inspect_patch(d("pyproject.toml"), INST)[0]
    assert "edits_tests" in inspect_patch(d("conftest.py"), INST)[0]
    assert "path_escape" in inspect_patch(d("../etc/passwd"), INST)[0]


def test_added_line_patterns():
    f, v = inspect_patch(d("pkg/mod.py", "+    pytest.skip('x')"), INST)
    assert "pytest_skip" in f and v == "invalid"
    f, v = inspect_patch(d("pkg/mod.py", "+    except Exception: pass"), INST)
    assert "broad_except_pass" in f and v == "ok"  # soft flag: logged, not gated
    f, v = inspect_patch(d("pkg/mod.py", "+    def __eq__(self, other): return True"), INST)
    assert "always_true_eq" in f


def test_quoted_paths_and_unparseable_headers():
    q = 'diff --git "a/tests/test_a.py" "b/tests/test_a.py"\n--- "a/tests/test_a.py"\n+++ "b/tests/test_a.py"\n@@ -1,1 +1,2 @@\n a\n+x = 1\n'
    f, v = inspect_patch(q, INST)
    assert v == "invalid" and "edits_tests" in f
    weird = "diff --git q/pkg/mod.py r/pkg/mod.py\n--- q/pkg/mod.py\n+++ r/pkg/mod.py\n@@ -1,1 +1,2 @@\n a\n+x = 1\n"
    f, v = inspect_patch(weird, INST)
    assert v == "invalid" and "path_escape" in f
    assert "edits_config" in inspect_patch(d("tox.ini"), INST)[0] and "edits_config" in inspect_patch(d("noxfile.py"), INST)[0]
