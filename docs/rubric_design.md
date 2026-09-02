# Rubric design (code profile)

Machine-readable rubric versions live in `rubrics/*.yaml` (schema: rubric_id, version, applicable_task_types,
components, weights, gating_condition, partial_score_policy, regression_policy, rusca_stage, rationale).
Per-instance criteria are the ones shipped in `extra_info.rubrics` (same schema as the T01 math rubric:
`{criterion, points, tags{judge, inject, scaffold_rank, scaffold_kind, verify, reward}}`).

## Criteria (code profile, from the data)
| verify | rank | points | task | scaffold text (example) | verifier signal |
|---|---|---|---|---|---|
| failing_test | 0 | 2.0 | swe | "Make these currently failing tests pass: test_a, test_b." | F2P pass fraction |
| impl_hint | 0 | 2.0 | ut | "Define exactly these names: …" | – (not checkable) |
| approach | 1 | 1.5 | all | "Locate the root cause first, then make the smallest edit that fixes it." | – |
| edge_cases | 2 | 1.5 | all | "Handle the boundary conditions the issue describes…" | – |
| complexity | 2 | 1.0 | ut | "Keep the solution within the stated time limit…" | no per-test timeout |
| tests_pass | 3 | 1.0 | swe | "Do not break any test that currently passes." | no observed P2P regression |
| output_format | 4 | 1.0 | all | "Reply with one unified diff inside a ```diff block." | extraction path == requested |
| no_extra | 4 | 0.5 | all | "Change only what the issue requires; no refactoring…" | touched ⊆ gold-touched ∧ size ≤ 3× gold |

Two roles, kept separate:
* **Scaffolding** (RUSCA): criteria injected into the rollout prompt by rank, count from `lam(step)` ×
  intra-group decay; removed by ≈24 % of training. Reward unaffected (rubric_weight 0) in the baseline.
* **Rubric reward** (ablation): only checkable criteria are scored (`verifier/rubric.py`); unverifiable ones
  are excluded from the denominator rather than judged by an LLM (no judge → no judge-hacking surface).

Versions: `rubric_v001_code_hint` (scaffold-only baseline), `rubric_v002_checkable` (checkable criteria enter
the reward with weight 0.2 during scaffold/transition). The final rubric is fixed after the ablation
(`rubrics/rubric_final.yaml`).
