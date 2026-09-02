# Ablation report

Each arm changes ONE variable relative to `exp_D000_baseline` (same data, seed, steps, GPU). Validation metrics are the verifier's components on the repo-disjoint validation split (`score/mean` = mean final reward, `rule_correctness` = complete solve rate).

## Offline verifier ablation — `synthetic_v1`

Gold-patch recovery rate per parser arm (mean over 9 response wrappings): {"baseline/last": 0.75, "fallback/concat_distinct": 0.838, "fallback/first": 0.776, "fallback/last": 0.838, "markdown/last": 0.442, "strict/last": 0.204}

## Offline verifier ablation — `synthetic_v2`

Gold-patch recovery rate per parser arm (mean over 9 response wrappings): {"baseline/last": 0.748, "fallback/concat_distinct": 0.887, "fallback/first": 0.776, "fallback/last": 0.887, "markdown/last": 0.443, "strict/last": 0.222}

