import pytest
from verifier.diff_parser import (PARSERS, extract, extract_code_block, validate_unified_diff,
                                  search_replace_to_diff, touched_paths)
from verifier.schemas import ExtractStatus

GOOD = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,4 +1,4 @@
 import os
-def f(x):
-    return x
+def f(x):
+    return x + 1
 # end
"""
GOOD2 = """diff --git a/pkg/other.py b/pkg/other.py
--- a/pkg/other.py
+++ b/pkg/other.py
@@ -1,2 +1,2 @@
-a = 1
+a = 2
 b = 3
"""
BAD_COUNT = GOOD.replace("@@ -1,4 +1,4 @@", "@@ -1,5 +1,4 @@")
BAD_BODY = GOOD.replace(" # end", "# end")  # context line without leading space -> corrupt


def fence(p, lang="diff"):
    return f"Here is my fix:\n```{lang}\n{p}```\nHope this helps."


@pytest.mark.parametrize("parser", ["markdown", "fallback", "baseline"])
def test_fenced_diff_extracted(parser):
    r = extract(fence(GOOD), parser)
    assert r.status == ExtractStatus.DIFF_PARSE_SUCCESS and r.fmt == "fenced_diff"
    assert r.patch == GOOD and r.touched_files == ("pkg/mod.py",)


def test_strict_rejects_prose_but_accepts_bare_or_single_fence():
    assert extract(fence(GOOD), "strict").status == ExtractStatus.DIFF_PARSE_FAIL
    assert extract(GOOD, "strict").status == ExtractStatus.DIFF_PARSE_SUCCESS
    assert extract("```diff\n" + GOOD + "```", "strict").status == ExtractStatus.DIFF_PARSE_SUCCESS
    assert extract(GOOD + "\nthanks\n", "strict").status == ExtractStatus.DIFF_PARSE_FAIL


def test_empty_and_prose_only_is_no_patch():
    for parser in PARSERS:
        assert extract("", parser).status == ExtractStatus.NO_PATCH
        assert extract("I think the bug is in mod.py but I am not sure.", parser).status == ExtractStatus.NO_PATCH


@pytest.mark.parametrize("parser", ["strict", "markdown", "fallback"])
def test_malformed_hunk_is_parse_fail_not_repaired(parser):
    for bad in (BAD_COUNT, BAD_BODY):
        r = extract("```diff\n" + bad + "```", parser)
        assert r.status == ExtractStatus.DIFF_PARSE_FAIL, (parser, r.note)
        # the returned patch is never a "fixed" version
        assert r.patch in ("", bad)


def test_baseline_does_not_validate():
    r = extract("```diff\n" + BAD_COUNT + "```", "baseline")
    assert r.status == ExtractStatus.DIFF_PARSE_SUCCESS  # documented weakness of the baseline cascade


def test_last_candidate_wins_by_default_and_concat_distinct_merges():
    text = fence(GOOD2) + "\nActually the final version:\n" + fence(GOOD)
    assert extract(text, "markdown").patch == GOOD
    assert extract(text, "markdown", select="first").patch == GOOD2
    merged = extract(text, "markdown", select="concat_distinct")
    assert merged.status == ExtractStatus.DIFF_PARSE_SUCCESS
    assert set(merged.touched_files) == {"pkg/mod.py", "pkg/other.py"}


def test_fallback_raw_git_trims_trailing_prose_at_hunk_boundary():
    r = extract("Fix:\n" + GOOD + "\nThis change increments x.\nMore words.\n", "fallback")
    assert r.status == ExtractStatus.DIFF_PARSE_SUCCESS and r.fmt == "raw_git"
    assert r.patch == GOOD
    # markdown parser refuses unfenced diffs (format signal for the ablation)
    assert extract("Fix:\n" + GOOD, "markdown").status == ExtractStatus.DIFF_PARSE_FAIL


def test_fenced_any_language():
    r = extract(fence(GOOD, "python"), "fallback")
    assert r.status == ExtractStatus.DIFF_PARSE_SUCCESS and r.fmt == "fenced_any"


def test_search_replace_conversion_exact_unique_only():
    src = "import os\ndef f(x):\n    return x\n# end\n"
    files = {"pkg/mod.py": src}
    sr = "<solution>\n### pkg/mod.py\n<<<<<<< SEARCH\n    return x\n=======\n    return x + 1\n>>>>>>> REPLACE\n</solution>"
    d, note = search_replace_to_diff(sr, files)
    ok, reason, _ = validate_unified_diff(d)
    assert ok, reason
    assert "+    return x + 1" in d and touched_paths(d) == ("pkg/mod.py",)
    # ambiguous (two matches) -> not converted
    sr2 = sr.replace("    return x\n=======", "x\n=======")
    d2, note2 = search_replace_to_diff(sr2, {"pkg/mod.py": "x\nx\n"})
    assert d2 == "" and "not unique" in note2
    # no file content -> no synthesis (never invent line numbers)
    assert search_replace_to_diff(sr, None)[0] == ""
    r = extract(sr, "fallback", buggy_files=files)
    assert r.status == ExtractStatus.DIFF_PARSE_SUCCESS and r.fmt == "search_replace"
    r = extract(sr, "fallback", buggy_files=files, allow_search_replace=False)
    assert r.status == ExtractStatus.NO_PATCH


def test_begin_patch_only_in_baseline_and_flagged():
    bp = "*** Begin Patch\n*** Update File: pkg/mod.py\n@@\n-    return x\n+    return x + 1\n*** End Patch\n"
    assert extract(bp, "fallback").status == ExtractStatus.NO_PATCH
    r = extract(bp, "baseline")
    assert r.status == ExtractStatus.DIFF_PARSE_SUCCESS and r.fmt == "begin_patch" and "repair_risk" in r.note


def test_crlf_normalised():
    r = extract(fence(GOOD.replace("\n", "\r\n")), "markdown")
    assert r.status == ExtractStatus.DIFF_PARSE_SUCCESS and "\r" not in r.patch


def test_new_file_and_deleted_file_headers_validate():
    p = "diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n"
    assert validate_unified_diff(p)[0]
    d = "diff --git a/old.py b/old.py\ndeleted file mode 100644\n--- a/old.py\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-x = 1\n"
    assert validate_unified_diff(d)[0]


def test_no_newline_marker_ok():
    p = GOOD.rstrip("\n") + "\n\\ No newline at end of file\n"
    assert validate_unified_diff(p)[0]


def test_code_block_extraction():
    r = extract_code_block("Sure:\n```python\ndef f(x):\n    return x\n```\n")
    assert r.status == ExtractStatus.DIFF_PARSE_SUCCESS and r.patch.startswith("def f")
    r = extract_code_block("```python\ndef f(x:\n    return x\n```")
    assert r.status == ExtractStatus.DIFF_PARSE_FAIL and "SyntaxError" in r.note
    assert extract_code_block("no code here").status == ExtractStatus.NO_PATCH
    # last block wins
    r = extract_code_block("```python\nx=1\n```\n```python\nx=2\n```")
    assert r.patch.strip() == "x=2"


def test_leading_mail_headers_are_ignored_like_git_apply():
    text = "```diff\nFrom: bot <b@x>\nSubject: [PATCH] fix\n\n---\n" + GOOD + "```"
    r = extract(text, "markdown")
    assert r.status == ExtractStatus.DIFF_PARSE_SUCCESS and r.patch == GOOD
