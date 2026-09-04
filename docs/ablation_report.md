# Ablation report

Each arm changes ONE variable relative to `exp_D000_baseline` (same data, seed, steps, GPU). Validation metrics are the verifier's components on the repo-disjoint validation split (`score/mean` = mean final reward, `rule_correctness` = complete solve rate).

## Offline verifier ablation — `model_train`

Gold-patch recovery rate per parser arm (mean over 9 response wrappings): {"baseline/last": 0.742, "fallback/concat_distinct": 0.882, "fallback/first": 0.771, "fallback/last": 0.882, "markdown/last": 0.438, "strict/last": 0.222}

On real policy samples: {"baseline/last": {"extract_rate": 0.6662995594713657, "valid_rate": 0.6662995594713657, "n": 1816}, "fallback/concat_distinct": {"extract_rate": 0.6795154185022027, "valid_rate": 0.12555066079295155, "n": 1816}, "fallback/first": {"extract_rate": 0.6795154185022027, "valid_rate": 0.1299559471365639, "n": 1816}, "fallback/last": {"extract_rate": 0.6789647577092511, "valid_rate": 0.1222466960352423, "n": 1816}, "markdown/last": {"extract_rate": 0.6789647577092511, "valid_rate": 0.08975770925110133, "n": 1816}, "strict/last": {"extract_rate": 0.6178414096916299, "valid_rate": 0.0022026431718061676, "n": 1816}}

## Offline verifier ablation — `model_validation`

Gold-patch recovery rate per parser arm (mean over 9 response wrappings): {"baseline/last": 0.742, "fallback/concat_distinct": 0.882, "fallback/first": 0.771, "fallback/last": 0.882, "markdown/last": 0.438, "strict/last": 0.222}

On real policy samples: {"baseline/last": {"extract_rate": 0.6035714285714285, "valid_rate": 0.6035714285714285, "n": 280}, "fallback/concat_distinct": {"extract_rate": 0.6214285714285714, "valid_rate": 0.08214285714285714, "n": 280}, "fallback/first": {"extract_rate": 0.6214285714285714, "valid_rate": 0.08214285714285714, "n": 280}, "fallback/last": {"extract_rate": 0.6178571428571429, "valid_rate": 0.075, "n": 280}, "markdown/last": {"extract_rate": 0.6178571428571429, "valid_rate": 0.05, "n": 280}, "strict/last": {"extract_rate": 0.49642857142857144, "valid_rate": 0.0035714285714285713, "n": 280}}

## Offline verifier ablation — `synthetic_v1`

Gold-patch recovery rate per parser arm (mean over 9 response wrappings): {"baseline/last": 0.75, "fallback/concat_distinct": 0.838, "fallback/first": 0.776, "fallback/last": 0.838, "markdown/last": 0.442, "strict/last": 0.204}

## Offline verifier ablation — `synthetic_v2`

Gold-patch recovery rate per parser arm (mean over 9 response wrappings): {"baseline/last": 0.748, "fallback/concat_distinct": 0.887, "fallback/first": 0.776, "fallback/last": 0.887, "markdown/last": 0.443, "strict/last": 0.222}

## F2P partial credit

| experiment_id       | reward_version     |   val_score |   delta_val_score |   val_resolved |   delta_val_resolved |   val_resolved_swe |   val_resolved_ut |    val_f2p |   val_p2p |   val_apply |   val_format |   train_reward |   kl |   max_grad_norm | decision   |
|:--------------------|:-------------------|------------:|------------------:|---------------:|---------------------:|-------------------:|------------------:|-----------:|----------:|------------:|-------------:|---------------:|-----:|----------------:|:-----------|
| exp_D000_baseline   | rw_v001_baseline   |    0.213019 |            0      |        0.12625 |               0      |                  0 |            0.2525 |   0.219103 |         1 |     0.49625 |     0.681964 |       0.217279 |    0 |        0.384766 | keep       |
| exp_D001_f2p_binary | rw_v002_f2p_binary |    0.176964 |           -0.0361 |        0.13    |               0.0038 |                  0 |            0.26   |   0.211066 |         1 |     0.4925  |     0.721071 |       0.221528 |    0 |        0.472656 | keep       |
| exp_D002_f2p_linear | rw_v003_f2p_linear |    0.237196 |            0.0242 |        0.14375 |               0.0175 |                  0 |            0.2875 |   0.222113 |         1 |     0.49375 |     0.736607 |       0.288328 |    0 |        0.429688 | keep       |
| exp_D003_f2p_ladder | rw_v005_f2p_ladder |  nan        |          nan      |      nan       |             nan      |                nan |          nan      | nan        |       nan |   nan       |   nan        |     nan        |  nan |        0        | reject     |

## P2P regression penalty

| experiment_id         | reward_version       |   val_score |   delta_val_score |   val_resolved |   delta_val_resolved |   val_resolved_swe |   val_resolved_ut |    val_f2p |   val_p2p |   val_apply |   val_format |   train_reward |   kl |   max_grad_norm | decision   |
|:----------------------|:---------------------|------------:|------------------:|---------------:|---------------------:|-------------------:|------------------:|-----------:|----------:|------------:|-------------:|---------------:|-----:|----------------:|:-----------|
| exp_D000_baseline     | rw_v001_baseline     |    0.213019 |                 0 |        0.12625 |                    0 |                  0 |            0.2525 |   0.219103 |         1 |     0.49625 |     0.681964 |       0.217279 |    0 |        0.384766 | keep       |
| exp_D004_p2p_strong   | rw_v006_p2p_strong   |  nan        |               nan |      nan       |                  nan |                nan |          nan      | nan        |       nan |   nan       |   nan        |     nan        |  nan |        0        | reject     |
| exp_D005_p2p_tolerant | rw_v007_p2p_tolerant |  nan        |               nan |      nan       |                  nan |                nan |          nan      | nan        |       nan |   nan       |   nan        |     nan        |  nan |        0        | reject     |

## Rule-only vs rubric vs RUSCA

| experiment_id         | reward_version       |   val_score |   delta_val_score |   val_resolved |   delta_val_resolved |   val_resolved_swe |   val_resolved_ut |    val_f2p |   val_p2p |   val_apply |   val_format |   train_reward |   kl |   max_grad_norm | decision   |
|:----------------------|:---------------------|------------:|------------------:|---------------:|---------------------:|-------------------:|------------------:|-----------:|----------:|------------:|-------------:|---------------:|-----:|----------------:|:-----------|
| exp_D000_baseline     | rw_v001_baseline     |    0.213019 |                 0 |        0.12625 |                    0 |                  0 |            0.2525 |   0.219103 |         1 |     0.49625 |     0.681964 |       0.217279 |    0 |        0.384766 | keep       |
| exp_D006_rule_only    | rw_v011_rule_only    |  nan        |               nan |      nan       |                  nan |                nan |          nan      | nan        |       nan |   nan       |   nan        |     nan        |  nan |        0        | reject     |
| exp_D007_rubric_heavy | rw_v012_rubric_heavy |  nan        |               nan |      nan       |                  nan |                nan |          nan      | nan        |       nan |   nan       |   nan        |     nan        |  nan |        0        | reject     |
| exp_D008_rusca        | rw_v013_rusca        |  nan        |               nan |      nan       |                  nan |                nan |          nan      | nan        |       nan |   nan       |   nan        |     nan        |  nan |        0        | reject     |
| exp_D009_no_scaffold  | rw_v001_baseline     |  nan        |               nan |      nan       |                  nan |                nan |          nan      | nan        |       nan |   nan       |   nan        |     nan        |  nan |      nan        | reject     |

## Gating

| experiment_id            | reward_version          |   val_score |   delta_val_score |   val_resolved |   delta_val_resolved |   val_resolved_swe |   val_resolved_ut |    val_f2p |   val_p2p |   val_apply |   val_format |   train_reward |   kl |   max_grad_norm | decision   |
|:-------------------------|:------------------------|------------:|------------------:|---------------:|---------------------:|-------------------:|------------------:|-----------:|----------:|------------:|-------------:|---------------:|-----:|----------------:|:-----------|
| exp_D000_baseline        | rw_v001_baseline        |    0.213019 |                 0 |        0.12625 |                    0 |                  0 |            0.2525 |   0.219103 |         1 |     0.49625 |     0.681964 |       0.217279 |    0 |        0.384766 | keep       |
| exp_D010_additive_nogate | rw_v014_additive_nogate |  nan        |               nan |      nan       |                  nan |                nan |          nan      | nan        |       nan |   nan       |   nan        |     nan        |  nan |        0        | reject     |

## Length (DDCA)

| experiment_id     | reward_version   |   val_score |   delta_val_score |   val_resolved |   delta_val_resolved |   val_resolved_swe |   val_resolved_ut |    val_f2p |   val_p2p |   val_apply |   val_format |   train_reward |   kl |   max_grad_norm | decision   |
|:------------------|:-----------------|------------:|------------------:|---------------:|---------------------:|-------------------:|------------------:|-----------:|----------:|------------:|-------------:|---------------:|-----:|----------------:|:-----------|
| exp_D000_baseline | rw_v001_baseline |    0.213019 |                 0 |        0.12625 |                    0 |                  0 |            0.2525 |   0.219103 |         1 |     0.49625 |     0.681964 |       0.217279 |    0 |        0.384766 | keep       |
| exp_D011_ddca     | rw_v001_baseline |  nan        |               nan |      nan       |                  nan |                nan |          nan      | nan        |       nan |   nan       |   nan        |     nan        |  nan |        0        | reject     |

## Verifier / parser

| experiment_id            | reward_version   |   val_score |   delta_val_score |   val_resolved |   delta_val_resolved |   val_resolved_swe |   val_resolved_ut |    val_f2p |   val_p2p |   val_apply |   val_format |   train_reward |   kl |   max_grad_norm | decision   |
|:-------------------------|:-----------------|------------:|------------------:|---------------:|---------------------:|-------------------:|------------------:|-----------:|----------:|------------:|-------------:|---------------:|-----:|----------------:|:-----------|
| exp_D000_baseline        | rw_v001_baseline |    0.213019 |                 0 |        0.12625 |                    0 |                  0 |            0.2525 |   0.219103 |         1 |     0.49625 |     0.681964 |       0.217279 |    0 |        0.384766 | keep       |
| exp_D012_parser_baseline | rw_v001_baseline |  nan        |               nan |      nan       |                  nan |                nan |          nan      | nan        |       nan |   nan       |   nan        |     nan        |  nan |        0        | reject     |

