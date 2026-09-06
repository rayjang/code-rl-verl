#!/bin/bash
# Download the 20 Singularity images (~23 GB) from the GitHub Release of this repo into
# environments/sif/. Idempotent: an existing non-empty .sif of the right size is skipped.
#
#   GITHUB_TOKEN=<pat> bash scripts/fetch_sif_release.sh        # private repo needs a token
#
# Alternative that needs no token and no release: scripts/pull_images.sh rebuilds the exact
# same images straight from Docker Hub (public jyangballin/swesmith.x86_64.* tags).
set -uo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DEST=${RL_SIF_DIR:-$ROOT/environments/sif}
REPO=${SIF_REPO:-rayjang/code-rl-verl}
TAG=${SIF_TAG:-sif-v1}
: "${GITHUB_TOKEN:?set GITHUB_TOKEN to a PAT with repo scope}"
mkdir -p "$DEST"
AUTH=(-H "Authorization: token $GITHUB_TOKEN")

api="https://api.github.com/repos/$REPO/releases/tags/$TAG"
assets=$(curl -sSL "${AUTH[@]}" "$api" | python3 -c '
import json,sys
r=json.load(sys.stdin)
if "assets" not in r: sys.exit(f"release not found: {r.get(\"message\")}")
for a in r["assets"]: print(a["id"], a["size"], a["name"])
') || exit 1

echo "$assets" | while read -r id size name; do
  out="$DEST/$name"
  if [[ -f "$out" && "$(stat -c%s "$out")" == "$size" ]]; then echo "skip $name"; continue; fi
  echo "[$(date -Is)] get $name ($((size/1024/1024)) MB)"
  curl -sSL -o "$out" "${AUTH[@]}" -H "Accept: application/octet-stream" \
    "https://api.github.com/repos/$REPO/releases/assets/$id" || { echo "FAIL $name"; rm -f "$out"; }
done
ls -la "$DEST"/*.sif 2>/dev/null | awk '{print $5, $9}'; du -sh "$DEST"
