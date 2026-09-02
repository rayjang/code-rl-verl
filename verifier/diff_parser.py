"""Patch extraction candidates.

All parsers share the same contract:

    parse(text, *, buggy_files=None) -> ExtractionResult

Allowed normalisation (does NOT change patch semantics): CRLF->LF, stripping the markdown fence,
guaranteeing a trailing newline, dropping trailing non-diff prose *after the last hunk ends*.
Forbidden (would be an auto-repair and therefore a reward-hacking vector): fixing hunk line
counts, inventing headers or line numbers, fuzzy path matching, fuzz application.

Candidates
  strict        the whole response (minus one optional ```diff fence) must be a valid unified diff
  markdown      last ```diff / ```patch fenced block, validated
  fallback      markdown -> any fence with a diff signature -> raw `diff --git`/`--- a/` region
                (conservative: trailing prose trimmed only at a hunk boundary), validated
  baseline      port of rl_code_v1/reward/patch_extract.py 6-stage cascade (NOT validated,
                includes the `*** Begin Patch` conversion that fabricates hunk headers -> kept only
                as an ablation candidate and flagged `repair_risk`)

SEARCH/REPLACE blocks (the format the dataset's *system* prompt asks for) are converted to a
unified diff only when `buggy_files` is given and every SEARCH matches exactly once; that is a
format conversion of a valid edit, not a repair, and it is reported as fmt="search_replace".
"""
from __future__ import annotations

import difflib
import re
from typing import Callable, Optional

from .schemas import ExtractStatus, ExtractionResult

FENCE_RE = re.compile(r"```[ \t]*([\w+.-]*)[ \t]*\r?\n(.*?)```", re.S)
DIFF_SIG_RE = re.compile(r"^(diff --git |--- |\+\+\+ |@@ )", re.M)
GIT_HDR_RE = re.compile(r"^diff --git ", re.M)
UNI_HDR_RE = re.compile(r"^--- (?:a/|/dev/null)", re.M)
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
BEGIN_PATCH_RE = re.compile(r"\*\*\* Begin Patch\r?\n(.*?)\*\*\* End Patch", re.S)
SR_RE = re.compile(
    r"###[ \t]*(?P<path>[^\n]+?)[ \t]*\r?\n"
    r"<<<<<<<[ \t]*SEARCH[ \t]*\r?\n(?P<search>.*?)\r?\n"
    r"=======[ \t]*\r?\n(?P<replace>.*?)\r?\n"
    r">>>>>>>[ \t]*REPLACE", re.S)
DIFF_PATH_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)


def normalize(p: str) -> str:
    """CRLF -> LF, strip leading/trailing blank lines, guarantee trailing newline."""
    p = (p or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    return (p + "\n") if p else ""


def touched_paths(patch: str) -> tuple:
    out = []
    for m in DIFF_PATH_RE.finditer(patch or ""):
        out += [m.group(1), m.group(2)]
    for line in (patch or "").split("\n"):
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            out.append(line[6:].strip().split("\t")[0])
    return tuple(sorted({p for p in out if p and p != "/dev/null"}))


# ----------------------------------------------------------------------------------------------
# structural validation (mirrors what `git apply` rejects as "corrupt patch")
# ----------------------------------------------------------------------------------------------
def validate_unified_diff(patch: str) -> tuple[bool, str, int]:
    """Return (valid, reason, end_index_of_last_hunk_line).

    Rules: at least one file section with a `+++ ` header (or `diff --git` with a hunk);
    every hunk header's counts must equal the number of old/new lines in its body;
    a hunk body line must start with ' ', '+', '-' or '\\'. Trailing prose after the last
    complete hunk is reported through end_index so callers may trim it (never repair)."""
    lines = patch.split("\n")
    i, n = 0, len(lines)
    n_hunks, n_files, last_end = 0, 0, -1
    saw_header = False
    while i < n:
        line = lines[i]
        if line.startswith("diff --git ") or line.startswith("+++ "):
            saw_header = True
            if line.startswith("+++ "):
                n_files += 1
            i += 1
            continue
        if line.startswith("--- ") or line.startswith("index ") or line.startswith("new file mode") \
                or line.startswith("deleted file mode") or line.startswith("old mode") or line.startswith("new mode") \
                or line.startswith("similarity index") or line.startswith("rename ") or line.startswith("copy ") \
                or line.startswith("Binary files") or line.startswith("GIT binary patch"):
            i += 1
            continue
        m = HUNK_RE.match(line)
        if m:
            if not saw_header:
                return False, "hunk before any file header", -1
            old_n = int(m.group(2)) if m.group(2) is not None else 1
            new_n = int(m.group(4)) if m.group(4) is not None else 1
            i += 1
            o = a = 0
            while i < n and (o < old_n or a < new_n):
                b = lines[i]
                if b.startswith("\\"):
                    i += 1
                    continue
                if b == "" and i == n - 1:
                    break
                c = b[:1] if b else " "
                if c == " " or b == "":
                    o += 1; a += 1
                elif c == "-":
                    o += 1
                elif c == "+":
                    a += 1
                else:
                    return False, f"bad hunk body line {i + 1}: {b[:40]!r}", -1
                i += 1
            if o != old_n or a != new_n:
                return False, f"hunk count mismatch (header -{old_n} +{new_n}, body -{o} +{a})", -1
            while i < n and lines[i].startswith("\\"):
                i += 1
            n_hunks += 1
            last_end = i
            continue
        if not line.strip():
            i += 1
            continue
        # non-diff text inside the diff region
        if n_hunks == 0:
            return False, f"unexpected line before first hunk: {line[:40]!r}", -1
        # prose after a complete hunk: legal only as trailing garbage -> stop here
        break
    if n_hunks == 0:
        return False, "no hunks", -1
    return True, "", last_end


def _trim_to_last_hunk(patch: str) -> str:
    ok, _, end = validate_unified_diff(patch)
    if ok and end >= 0:
        return "\n".join(patch.split("\n")[:end])
    return patch


def _result(patch: str, fmt: str, parser: str, n_cand: int, note: str = "") -> ExtractionResult:
    patch = normalize(patch)
    if not patch:
        return ExtractionResult(ExtractStatus.NO_PATCH, "", "none", parser, n_cand, note or "empty")
    ok, reason, _ = validate_unified_diff(patch)
    if not ok:
        return ExtractionResult(ExtractStatus.DIFF_PARSE_FAIL, patch, fmt, parser, n_cand, reason, touched_paths(patch))
    return ExtractionResult(ExtractStatus.DIFF_PARSE_SUCCESS, patch, fmt, parser, n_cand, note, touched_paths(patch))


def _select(cands: list, policy: str) -> str:
    if not cands:
        return ""
    if policy == "first":
        return cands[0]
    if policy == "longest":
        return max(cands, key=len)
    if policy == "concat_distinct":
        seen, out = set(), []
        for c in cands:
            paths = set(touched_paths(c))
            if paths & seen:
                out = [c]          # a later block re-edits an earlier file -> it supersedes everything
                seen = paths
            else:
                out.append(c); seen |= paths
        return "\n".join(x.strip("\n") for x in out)
    return cands[-1]              # "last" (baseline rationale: final answer comes last)


# ----------------------------------------------------------------------------------------------
# SEARCH/REPLACE -> unified diff (format conversion; exact unique match required, no fuzz)
# ----------------------------------------------------------------------------------------------
def search_replace_to_diff(text: str, buggy_files: Optional[dict]) -> tuple[str, str]:
    """Return (diff, note). Empty diff when not convertible."""
    if not buggy_files:
        return "", "no buggy_files"
    edits: dict = {}
    for m in SR_RE.finditer(text):
        edits.setdefault(m.group("path").strip(), []).append((m.group("search"), m.group("replace")))
    if not edits:
        return "", "no SEARCH/REPLACE blocks"
    chunks, notes = [], []
    for path, pairs in edits.items():
        src = buggy_files.get(path)
        if src is None:
            base = path.split("/")[-1]
            cand = [k for k in buggy_files if k.endswith("/" + base) or k == base]
            if len(cand) != 1:
                notes.append(f"unknown path {path}"); continue
            path, src = cand[0], buggy_files[cand[0]]
        cur = src
        applied = 0
        for s, r in pairs:
            if cur.count(s) != 1:
                notes.append(f"{path}: SEARCH not unique/absent"); continue
            cur = cur.replace(s, r, 1); applied += 1
        if applied == 0 or cur == src:
            continue
        body = "".join(difflib.unified_diff(src.splitlines(True), cur.splitlines(True),
                                            fromfile=f"a/{path}", tofile=f"b/{path}", n=3))
        if body:
            if not body.endswith("\n"):
                body += "\n"
            chunks.append(f"diff --git a/{path} b/{path}\n{body}")
    return "".join(chunks), ";".join(notes)


# ----------------------------------------------------------------------------------------------
# candidate parsers
# ----------------------------------------------------------------------------------------------
def parse_strict(text: str, *, buggy_files=None, select: str = "last", allow_search_replace: bool = False) -> ExtractionResult:
    t = (text or "").replace("\r\n", "\n").strip("\n")   # never strip spaces: blank context lines are " "
    if not t.strip():
        return ExtractionResult(ExtractStatus.NO_PATCH, "", "none", "strict", 0, "empty response")
    fmt = "raw"
    m = re.fullmatch(r"```[ \t]*(diff|patch)?[ \t]*\n(.*?)\n?```", t, re.S)
    if m:
        t, fmt = m.group(2), "fenced_diff"
    if allow_search_replace and SR_RE.search(t):
        d, note = search_replace_to_diff(t, buggy_files)
        return _result(d, "search_replace", "strict", 1, note) if d else \
            ExtractionResult(ExtractStatus.DIFF_PARSE_FAIL, "", "search_replace", "strict", 1, note)
    if not DIFF_SIG_RE.search(t):
        return ExtractionResult(ExtractStatus.NO_PATCH, "", "none", "strict", 0, "no diff signature")
    if not (t.startswith("diff --git ") or t.startswith("--- ")):
        return ExtractionResult(ExtractStatus.DIFF_PARSE_FAIL, "", fmt, "strict", 1, "prose before diff")
    ok, reason, end = validate_unified_diff(normalize(t))
    if ok and end < len(normalize(t).split("\n")) - 1 and "".join(normalize(t).split("\n")[end:]).strip():
        return ExtractionResult(ExtractStatus.DIFF_PARSE_FAIL, "", fmt, "strict", 1, "prose after diff")
    return _result(t, fmt, "strict", 1)


def parse_markdown(text: str, *, buggy_files=None, select: str = "last", allow_search_replace: bool = True) -> ExtractionResult:
    t = (text or "").replace("\r\n", "\n")
    if not t.strip():
        return ExtractionResult(ExtractStatus.NO_PATCH, "", "none", "markdown", 0, "empty response")
    fenced = [m.group(2) for m in FENCE_RE.finditer(t) if (m.group(1) or "").lower() in ("diff", "patch")]
    if fenced:
        return _result(_select(fenced, select), "fenced_diff", "markdown", len(fenced))
    if allow_search_replace and SR_RE.search(t):
        d, note = search_replace_to_diff(t, buggy_files)
        return _result(d, "search_replace", "markdown", 1, note) if d else \
            ExtractionResult(ExtractStatus.DIFF_PARSE_FAIL, "", "search_replace", "markdown", 1, note)
    if DIFF_SIG_RE.search(t):
        return ExtractionResult(ExtractStatus.DIFF_PARSE_FAIL, "", "unfenced", "markdown", 1, "diff outside a ```diff fence")
    return ExtractionResult(ExtractStatus.NO_PATCH, "", "none", "markdown", 0, "no fenced diff")


def parse_fallback(text: str, *, buggy_files=None, select: str = "last", allow_search_replace: bool = True) -> ExtractionResult:
    t = (text or "").replace("\r\n", "\n")
    if not t.strip():
        return ExtractionResult(ExtractStatus.NO_PATCH, "", "none", "fallback", 0, "empty response")
    fenced_diff, fenced_any = [], []
    for m in FENCE_RE.finditer(t):
        lang, body = (m.group(1) or "").lower(), m.group(2)
        if lang in ("diff", "patch"):
            fenced_diff.append(body)
        elif DIFF_SIG_RE.search(body):
            fenced_any.append(body)
    if fenced_diff:
        return _result(_select(fenced_diff, select), "fenced_diff", "fallback", len(fenced_diff))
    if fenced_any:
        return _result(_select(fenced_any, select), "fenced_any", "fallback", len(fenced_any))
    if allow_search_replace and SR_RE.search(t):
        d, note = search_replace_to_diff(t, buggy_files)
        return _result(d, "search_replace", "fallback", 1, note) if d else \
            ExtractionResult(ExtractStatus.DIFF_PARSE_FAIL, "", "search_replace", "fallback", 1, note)
    # unfenced diff: several `diff --git` headers are almost always ONE multi-file patch -> start at the
    # first header (the baseline's "last header" rule silently drops earlier files of multi-file patches)
    ms = list(GIT_HDR_RE.finditer(t))
    if ms:
        region = t[ms[0].start():]
        return _result(_trim_to_last_hunk(normalize(region)), "raw_git", "fallback", len(ms))
    ms = list(UNI_HDR_RE.finditer(t))
    if ms:
        region = t[ms[0].start():]
        return _result(_trim_to_last_hunk(normalize(region)), "raw_unified", "fallback", len(ms))
    if DIFF_SIG_RE.search(t):
        return ExtractionResult(ExtractStatus.DIFF_PARSE_FAIL, "", "fragment", "fallback", 1, "hunk without file header")
    return ExtractionResult(ExtractStatus.NO_PATCH, "", "none", "fallback", 0, "no patch-like text")


def _from_begin_patch(body: str) -> str:
    """Baseline conversion of `*** Update File:` blocks. Fabricates `@@ -0,0 +0,0 @@` headers and
    relies on `patch --fuzz=5` downstream: an auto-repair path. Kept for the ablation only."""
    out, cur = [], None
    for line in body.split("\n"):
        m = re.match(r"\*\*\* (Update|Add|Delete) File: (.+)$", line.strip())
        if m:
            cur = m.group(2).strip()
            out += [f"diff --git a/{cur} b/{cur}", f"--- a/{cur}", f"+++ b/{cur}"]
            continue
        if cur is None:
            continue
        out.append("@@ -0,0 +0,0 @@" if line.startswith("@@") else line)
    return "\n".join(out) if cur else ""


def parse_baseline(text: str, *, buggy_files=None, select: str = "last", allow_search_replace: bool = True) -> ExtractionResult:
    """Port of rl_code_v1/reward/patch_extract.py (no validation; last candidate wins)."""
    t = (text or "").replace("\r\n", "\n")
    if not t.strip():
        return ExtractionResult(ExtractStatus.NO_PATCH, "", "none", "baseline", 0)
    fenced_diff, fenced_any = [], []
    for m in FENCE_RE.finditer(t):
        lang, body = (m.group(1) or "").lower(), m.group(2)
        if lang in ("diff", "patch"):
            fenced_diff.append(body)
        elif DIFF_SIG_RE.search(body):
            fenced_any.append(body)

    def mk(p, fmt, note=""):
        p = normalize(p)
        return ExtractionResult(ExtractStatus.DIFF_PARSE_SUCCESS if p else ExtractStatus.NO_PATCH, p, fmt if p else "none",
                                "baseline", 1, note, touched_paths(p))
    if fenced_diff:
        return mk(fenced_diff[-1], "fenced_diff")
    if fenced_any:
        return mk(fenced_any[-1], "fenced_any")
    ms = list(GIT_HDR_RE.finditer(t))
    if ms:
        return mk(t[ms[-1].start():], "raw_git")
    ms = list(UNI_HDR_RE.finditer(t))
    if ms:
        return mk(t[ms[-1].start():], "raw_unified")
    ms = list(BEGIN_PATCH_RE.finditer(t))
    if ms:
        conv = _from_begin_patch(ms[-1].group(1))
        if conv.strip():
            return mk(conv, "begin_patch", "repair_risk: fabricated hunk headers")
    if allow_search_replace:
        conv, note = search_replace_to_diff(t, buggy_files)
        if conv.strip():
            return mk(conv, "search_replace", note)
    return ExtractionResult(ExtractStatus.NO_PATCH, "", "none", "baseline", 0)


PARSERS: dict[str, Callable[..., ExtractionResult]] = {
    "strict": parse_strict,
    "markdown": parse_markdown,
    "fallback": parse_fallback,
    "baseline": parse_baseline,
}


def extract(text: str, parser: str = "fallback", **kw) -> ExtractionResult:
    return PARSERS[parser](text, **kw)


# ----------------------------------------------------------------------------------------------
# unit-test track: python code block extraction (same statuses; "patch" holds the code)
# ----------------------------------------------------------------------------------------------
def extract_code_block(text: str, lang: str = "python", select: str = "last") -> ExtractionResult:
    """Return the fenced code block for the unit-test track. Syntax is checked with compile()
    so that `diff_parse_fail` means 'block found but not valid python' (format), analogous to a
    malformed diff. No repair is attempted."""
    t = (text or "").replace("\r\n", "\n")
    if not t.strip():
        return ExtractionResult(ExtractStatus.NO_PATCH, "", "none", "code_block", 0, "empty response")
    blocks = [m.group(2) for m in FENCE_RE.finditer(t) if (m.group(1) or "").lower() in (lang, "py", "python3", "")]
    typed = [m.group(2) for m in FENCE_RE.finditer(t) if (m.group(1) or "").lower() in (lang, "py", "python3")]
    cands = typed or blocks
    if not cands:
        return ExtractionResult(ExtractStatus.NO_PATCH, "", "none", "code_block", 0, "no code fence")
    code = _select(cands, select) if select != "concat_distinct" else cands[-1]
    code = code.replace("\r\n", "\n")
    try:
        compile(code, "<solution>", "exec")
    except SyntaxError as e:
        return ExtractionResult(ExtractStatus.DIFF_PARSE_FAIL, code, "fenced_code", "code_block", len(cands), f"SyntaxError: {e.msg} line {e.lineno}")
    return ExtractionResult(ExtractStatus.DIFF_PARSE_SUCCESS, code, "fenced_code", "code_block", len(cands))
