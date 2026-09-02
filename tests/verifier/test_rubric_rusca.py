import math
from verifier.rubric import parse_rubrics, evaluate_rubric
from verifier.rusca import RuscaConfig, lam, stage, n_inject, select_criteria, inject
from verifier.schemas import ExecutionResult, ErrKind, TestOutcome, ExtractionResult, ExtractStatus

RUB = [{"criterion": "Make these failing tests pass", "points": 2.0, "tags": {"scaffold_rank": 0, "verify": "failing_test", "inject": True}},
       {"criterion": "Locate root cause", "points": 1.5, "tags": {"scaffold_rank": 1, "verify": "approach", "inject": True}},
       {"criterion": "Handle boundaries", "points": 1.5, "tags": {"scaffold_rank": 2, "verify": "edge_cases", "inject": True}},
       {"criterion": "Do not break passing tests", "points": 1.0, "tags": {"scaffold_rank": 3, "verify": "tests_pass", "inject": True}},
       {"criterion": "Reply with one diff", "points": 1.0, "tags": {"scaffold_rank": 4, "verify": "output_format", "inject": True}},
       {"criterion": "Change only what is required", "points": 0.5, "tags": {"scaffold_rank": 4, "verify": "no_extra", "inject": True}}]


def test_rubric_eval_checkable_only():
    r = parse_rubrics(RUB)
    ext = ExtractionResult(ExtractStatus.DIFF_PARSE_SUCCESS, "x" * 100, "fenced_diff", "markdown", 1, "", ("a.py",))
    er = ExecutionResult(True, ErrKind.OK, "", TestOutcome(("t1",), ("t2",)), TestOutcome(("p1", "p2"), ()))
    ev = evaluate_rubric(r, ext, er, gold_touched=["a.py"], gold_patch_len=80)
    assert ev.checked == 4 and set(ev.details) == {"failing_test", "tests_pass", "output_format", "no_extra"}
    # (2*0.5 + 1*1 + 1*1 + 0.5*1) / 4.5
    assert math.isclose(ev.score, (1.0 + 1 + 1 + 0.5) / 4.5)
    ev2 = evaluate_rubric(r, ext, er, gold_touched=["b.py"], gold_patch_len=10)
    assert ev2.details["no_extra"] == 0.0


def test_lam_matches_documented_points():
    cfg = RuscaConfig(total_steps=50)
    assert lam(0, cfg) > 0.999
    assert math.isclose(lam(10, cfg), 0.5, abs_tol=1e-6)
    assert lam(12, cfg) < 0.01 and lam(50, cfg) == 0.0 or lam(50, cfg) < 1e-6
    assert stage(0, cfg) == "scaffold" and stage(6, cfg) == "transition" and stage(12, cfg) == "rule"
    cfg2 = RuscaConfig(total_steps=200)
    assert math.isclose(lam(40, cfg2), 0.5, abs_tol=1e-6)


def test_injection_counts_and_order():
    cfg = RuscaConfig(total_steps=50, group_jitter=0.0)
    assert n_inject(0, 6, cfg) == 6 and n_inject(10, 6, cfg) == 3 and n_inject(12, 6, cfg) == 0
    sel = select_criteria(parse_rubrics(RUB), 3)
    assert [c["tags"]["scaffold_rank"] for c in sel] == [0, 1, 2]
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    out = inject(msgs, sel)
    assert out[1]["content"].startswith("task") and "Make these failing tests pass" in out[1]["content"]
    assert msgs[1]["content"] == "task"  # original untouched
    assert inject(msgs, []) == msgs
