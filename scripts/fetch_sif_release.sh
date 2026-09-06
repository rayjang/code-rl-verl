#!/bin/bash
# Download the 20 Singularity images (~23 GB) from the GitHub Release of this repo into
# environments/sif/. Idempotent: an existing .sif of the right size is skipped.
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

list=$(curl -sSL -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/$REPO/releases/tags/$TAG" | python3 -c '
import json, sys
r = json.load(sys.stdin)
if "assets" not in r:
    sys.exit("release %s not found: %s" % (sys.argv[1], r.get("message", "?")))
for a in r["assets"]:
    print(a["id"], a["size"], a["name"])
' "$TAG") || exit 1

[[ -n "$list" ]] || { echo "release $TAG has no assets"; exit 1; }

rc=0
while read -r id size name; do
  [[ -n "${name:-}" ]] || continue
  out="$DEST/$name"
  if [[ -f "$out" && "$(stat -c%s "$out")" == "$size" ]]; then echo "skip $name"; continue; fi
  echo "[$(date -Is)] get $name ($((size / 1024 / 1024)) MB)"
  if curl -sSL -o "$out" -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/octet-stream" \
       "https://api.github.com/repos/$REPO/releases/assets/$id"; then
    got=$(stat -c%s "$out")
    if [[ "$got" != "$size" ]]; then
      echo "FAIL $name: got $got bytes, expected $size"; rm -f "$out"; rc=1
    fi
  else
    echo "FAIL $name: download error"; rm -f "$out"; rc=1
  fi
done <<< "$list"

ls -la "$DEST"/*.sif 2>/dev/null | awk '{print $5, $9}'
du -sh "$DEST"
exit $rc
