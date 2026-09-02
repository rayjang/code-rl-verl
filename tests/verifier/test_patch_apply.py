import os, subprocess, textwrap
import pytest
from verifier.patch_apply import apply_patch, git_apply_check
from verifier.schemas import ApplyStatus

SRC = "import os\ndef f(x):\n    return x\n# end\n"
GOOD = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,4 +1,4 @@
 import os
 def f(x):
-    return x
+    return x + 1
 # end
"""


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(SRC)
    return str(tmp_path)


def test_check_and_apply_strict(tree):
    assert git_apply_check(GOOD, tree).status == ApplyStatus.GIT_APPLY_CHECK_SUCCESS
    r = apply_patch(GOOD, tree)
    assert r.status == ApplyStatus.PATCH_APPLY_SUCCESS and r.touched_files == ("pkg/mod.py",)
    assert "return x + 1" in open(os.path.join(tree, "pkg/mod.py")).read()


def test_wrong_context_fails_strict_but_fuzz_rescues(tree):
    shifted = GOOD.replace(" import os\n", " import sys\n")   # context mismatch
    assert apply_patch(shifted, tree).status == ApplyStatus.PATCH_APPLY_FAIL
    assert "return x + 1" not in open(os.path.join(tree, "pkg/mod.py")).read()
    r = apply_patch(shifted, tree, mode="fuzz")
    assert r.status == ApplyStatus.PATCH_APPLY_SUCCESS and "repair path" in r.error


def test_wrong_path_fails(tree):
    r = apply_patch(GOOD.replace("pkg/mod.py", "pkg/nothere.py"), tree)
    assert r.status == ApplyStatus.PATCH_APPLY_FAIL


def test_corrupt_hunk_fails(tree):
    r = apply_patch(GOOD.replace("@@ -1,4 +1,4 @@", "@@ -1,9 +1,4 @@"), tree)
    assert r.status == ApplyStatus.PATCH_APPLY_FAIL


def test_reverse_apply(tree):
    apply_patch(GOOD, tree)
    r = apply_patch(GOOD, tree, reverse=True)
    assert r.status == ApplyStatus.PATCH_APPLY_SUCCESS
    assert open(os.path.join(tree, "pkg/mod.py")).read() == SRC


def test_empty_patch(tree):
    assert apply_patch("", tree).status == ApplyStatus.PATCH_APPLY_FAIL
