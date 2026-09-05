#!/usr/bin/env python3
"""Render DATASET.md (Korean) for a single-file difficulty-annotated dataset from its manifest + instance table."""
import json, os, sys, collections
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = sys.argv[1] if len(sys.argv) > 1 else "curated_v4_single"
D = f"{ROOT}/data/curated/{V}"
m = json.load(open(f"{D}/dataset_manifest.json")); t = pd.read_csv(f"{D}/instance_difficulty.csv")
ORDER = ["unsolved_no_signal", "very_hard", "hard", "medium", "easy", "very_easy", "trivial"]
DESC = {"unsolved_no_signal": "p̂=0 이고 8개 rollout의 reward 분산도 0 (형식/부분점수 신호조차 없음)", "very_hard": "0 ≤ p̂ < 0.05 (부분 신호 있음)", "hard": "0.05 ≤ p̂ < 0.2",
        "medium": "0.2 ≤ p̂ < 0.5", "easy": "0.5 ≤ p̂ < 0.8", "very_easy": "0.8 ≤ p̂ < 0.95", "trivial": "p̂ ≥ 0.95 (거의 항상 해결)"}
L = [f"# DATASET.md — `{V}` (난이도 표기 단일 데이터셋)\n",
     f"* 행 수: **{m['rows']:,}** (train/validation/test 구분 없음; 원래 split은 `extra_info.original_split` 에 정보로만 남김)",
     f"* 난이도 측정 모델: `{m['difficulty_model']}` — 각 인스턴스당 8개 rollout(temperature 1.0), verifier `vf_v002`, reward `rw_v001_baseline` 로 실행 채점",
     "* `empirical_success_rate` (p̂) = 8개 중 완전 해결(모든 effective F2P 통과 + P2P 회귀 없음) 비율. `reward_std` = rollout 간 reward 표준편차.",
     "* 난이도가 측정되지 않은 행(확장 unit-test 16,464행, R2E 120행, 환경 검증 실패 44행)은 포함하지 않았습니다.",
     "* 파일: `dataset.parquet` (verl 입력 형식) / `dataset.jsonl.gz` (동일 내용) / `instance_difficulty.csv` (요약표) / `dataset_manifest.json`\n",
     "## 1. 난이도 구간 정의 (`extra_info.difficulty`)\n", "| difficulty | 정의 | 행 수 |", "|---|---|---|"]
for d in ORDER:
    L.append(f"| {d} | {DESC[d]} | {m['by_difficulty'].get(d, 0):,} |")
L += ["\n`difficulty_score` = round(10·(1−p̂)) (0=항상 해결, 10=한 번도 해결 못 함). `train_recommended` = 0.05 ≤ p̂ ≤ 0.95 이거나 p̂<0.05 이지만 reward 분산>0 인 행 "
      f"(GRPO에서 그룹 내 분산이 생기는 행) — 총 **{m['train_recommended']:,}** 행.\n", "## 2. 태스크 × 난이도\n"]
tasks = sorted(m["by_task"])
L.append("| task_type | n | mean p̂ | " + " | ".join(ORDER) + " | train_recommended |"); L.append("|---|---|---|" + "---|" * len(ORDER) + "---|")
for tk in tasks:
    row = [str(m["by_task_difficulty"].get(f"{tk}|{d}", 0)) for d in ORDER]
    L.append(f"| {tk} | {m['by_task'][tk]['n']:,} | {m['by_task'][tk]['mean_p_hat']:.3f} | " + " | ".join(row) + f" | {m['by_task'][tk]['train_recommended']:,} |")
L += ["\n### 2.1 SWE-smith 컨텍스트 버킷별\n", "| variant_type | n | mean p̂ | p̂=0 | 0.2≤p̂≤0.8 | mean prompt tokens |", "|---|---|---|---|---|---|"]
for v, g in t[t["task_type"] == "swe_patch"].groupby("variant_type"):
    p = g["empirical_success_rate"]
    L.append(f"| {v} | {len(g)} | {p.mean():.3f} | {(p == 0).mean():.0%} | {((p >= 0.2) & (p <= 0.8)).mean():.0%} | {g['prompt_tokens'].mean():,.0f} |")
L += ["\n## 3. 태스크 설명\n",
      "| task_type | data_source | 입력 | 모델 출력 | 채점 |", "|---|---|---|---|---|",
      "| swe_patch | `t15_repo_patch` | GitHub 이슈 + 저장소 스냅샷(18k–119k 토큰) | unified diff(```` ```diff ````) 또는 `<solution>` SEARCH/REPLACE | 저장소 이미지 컨테이너에서 F2P 통과율·P2P 회귀 (`environments/`) |",
      "| unittest_function | `t15_unittest_impl` | 문제 설명(+테스트가 쓰는 이름) | ```` ```python ```` 코드 블록 | 숨겨진 assert 개별 실행(부분점수) |",
      "| unittest_pytest | `t15_unittest_impl` | 문제 설명(+import 이름) | solution.py 코드 블록 | 숨겨진 pytest 파일, 케이스별 통과 |",
      "| unittest_stdio | `t15_unittest_impl` | 입출력 형식 문제 | stdin→stdout 프로그램 | (stdin, stdout) 쌍 비교 |",
      "\n## 4. 컬럼 (verl 표준 5컬럼)\n",
      "`data_source`, `prompt`(chat list), `ability`, `reward_model{ground_truth=instance_id, style}`, `extra_info` — extra_info 주요 키:",
      "`instance_id`(채점 키), `task_type`, `variant_type`, `lang`, `repo`, `base_commit`, `branch`, `environment_id`, `rubrics`(scaffold 기준), "
      "`empirical_success_rate`, `reward_mean`, `reward_std`, `partial_signal`, `difficulty`, `difficulty_score`, `difficulty_model`, `difficulty_protocol`, "
      "`max_f2p_frac`, `apply_rate`, `original_split`, `train_recommended`, `quality_flags`, `n_f2p_effective`, `n_p2p_effective`, `prompt_tokens`, `verifier_version`, `reward_version`, `rubric_version`.\n",
      "## 5. 사용 예\n", "```python", "import pandas as pd", f"df = pd.read_parquet('data/{V}/dataset.parquet')",
      "ei = pd.DataFrame([dict(x) for x in df['extra_info']])", "train = df[ei['train_recommended'].values]                       # 학습 권장 밴드",
      "hard  = df[ei['difficulty'].isin(['hard','very_hard']).values]      # 난이도별 선택", "heldout = df[(ei['original_split']=='test').values]            # 원래의 저장소-분리 test split",
      "```", "split을 다시 나눌 때 SWE는 `extra_info.repo` 단위로 group split 하는 것을 권장합니다(같은 저장소의 인스턴스가 train/test에 동시에 들어가지 않게).\n",
      "## 6. 품질 플래그 (`extra_info.quality_flags`, `;` 구분)\n",
      "`PROMPT_FIX_ENTRY_POINT` 테스트가 요구하는 이름을 프롬프트에 덧붙임 · `GOLD_WHITESPACE_MISMATCH` gold가 `--ignore-whitespace`로만 적용 · `F2P_PRUNED:n`/`P2P_PRUNED:n` 환경에서 실패하는 테스트 id 제거 · "
      "`gold_touches_tests` gold가 테스트 패턴 파일도 수정 · `no_reference_solution` stdio에 참조 해답 없음(빈 코드로만 검증) · `EMPTY_PASSES_SOME:a/b` no-op 코드가 일부 테스트 통과.\n",
      "## 7. 원본 출처\n", "`rl_code_v1` (SWE-smith 751 → 환경 검증 통과 715; unit-test core 5,927 → 5,919: opencodeinstruct / kodcode / rstar-coder / ko-native). 원본 행은 수정하지 않았고 프롬프트에는 위 플래그의 변경만 있습니다."]
open(f"{D}/DATASET.md", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("wrote", f"{D}/DATASET.md", len(L), "lines")
