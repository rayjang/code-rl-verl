#!/bin/bash
#SBATCH -p koni -w gpu48 -c 8 --mem=48G -t 06:00:00 -J pull_sif
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/pull_sif_%j.out
# Pull the 19 SWE-smith repository images + the unittest sandbox base image as .sif
# CPU-only job on gpu48 (no GPUs requested). Idempotent: existing non-empty .sif are skipped.
set -uo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DEST=${RL_SIF_DIR:-$ROOT/environments/sif}
# image list ships with the repo; the sources/ tree is not versioned
LIST=$ROOT/environments/images/images_t15_code_swesmith.txt
[[ -f "$LIST" ]] || LIST=$ROOT/sources/rl_code_v1/data/index/images_t15_code_swesmith.txt
export SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR:-$HOME/.singularity/cache}
export SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR:-/tmp/sing_${USER}_${SLURM_JOB_ID:-$$}}
mkdir -p "$DEST" "$SINGULARITY_TMPDIR"
echo "host=$(hostname) start=$(date -Is) CVD=${CUDA_VISIBLE_DEVICES:-none}"
pull_one() {
  img="$1"; sif="$DEST/${img##*/}.sif"; sif="${sif//:/_}"
  if [[ -s "$sif" ]]; then echo "skip $sif"; return 0; fi
  echo "[$(date -Is)] pull $img -> $sif"
  if singularity pull --force "$sif" "docker://$img" > "$DEST/${img##*/}.pull.log" 2>&1; then
    echo "[$(date -Is)] ok $(du -h "$sif" | cut -f1) $sif"
  else
    echo "[$(date -Is)] FAIL $img (see ${img##*/}.pull.log)"; rm -f "$sif"; return 1
  fi
}
export -f pull_one; export DEST
# unittest sandbox base (no fakeroot available -> pull base image; pytest goes into a bind-mounted venv)
pull_one "python:3.11-slim-bookworm"
grep -v '^\s*$' "$LIST" | xargs -P 3 -I{} bash -c 'pull_one "$@"' _ {}
echo "done=$(date -Is)"; ls -la "$DEST"/*.sif | awk '{print $5, $9}'; du -sh "$DEST"
rm -rf "$SINGULARITY_TMPDIR"
