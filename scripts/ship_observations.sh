#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Ship an observation zip to the GitHub remote, so a session that cannot see
# this machine can fetch it with git. A chat upload does not carry a file this
# size: a 234 MB collector zip once arrived as its first 26.75 MiB.
# ============================================================================
#   scripts/ship_observations.sh outputs/obs_data/<run>.zip
#
# 1. Checks the zip: every member's CRC, then prints its dataset, sample count
#    and read_gate verdict (anything but `gated` is not a gated run).
# 2. Writes <run>.zip.sha256 and splits the zip into 45 MB parts (GitHub
#    refuses files over 100 MB).
# 3. Commits them on a throwaway ORPHAN branch obs-data/<run>, from a scratch
#    repo next to the zip (your checkout is untouched), pushing 10 parts
#    (~450 MB) at a time (GitHub caps one push at 2 GB). The scratch repo is
#    removed once everything is pushed.
#
# Needs push access to origin from THIS machine. A public repo pulls without a
# login; a push always needs one. Check first (writes nothing):
#   git push --dry-run origin HEAD:refs/heads/obs-data/push-test
# On a cluster, run it where there is internet (a login node), not in the job.
#
# Re-running ships from scratch and overwrites the branch. Once the data has
# been fetched, delete it:  git push origin --delete obs-data/<run>
#
# Receiving side: fetch the branch, `cat <run>.zip.part* > <run>.zip`,
# `sha256sum -c <run>.zip.sha256`.
# ============================================================================

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <observation zip>" >&2
    exit 2
fi
Z=$(realpath "$1")
[[ -f "$Z" ]] || { echo "error: $Z not found" >&2; exit 1; }
RUN=$(basename "$Z" .zip)
PART=${PART:-45m}
BATCH=${BATCH:-10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
REMOTE=$(git remote get-url origin)
NAME=$(git config user.name || true)
EMAIL=$(git config user.email || true)
if [[ -z "$NAME" || -z "$EMAIL" ]]; then
    echo "error: set git config --global user.name / user.email first" >&2
    exit 1
fi

# 1. `python -m zipfile -t` prints a corrupt member but still exits 0, so test here.
ls -l "$Z"
python - "$Z" <<'PY'
import json, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    bad = zf.testzip()
    if bad is not None:
        sys.exit(f"error: {bad} is corrupt in {sys.argv[1]}")
    meta = json.loads(zf.read("meta.json"))
ds = (meta.get("dataset_resolved") or {}).get("corpus")
verdict = (meta.get("read_gate") or {}).get("verdict")
print(f"zip OK | dataset: {ds} | samples: {len(meta['samples'])} | read_gate: {verdict}")
PY

# 2. Checksum + parts, in a scratch repo next to the zip.
SHIP="$(dirname "$Z")/ship_$RUN"
rm -rf "$SHIP"
mkdir -p "$SHIP"
(cd "$(dirname "$Z")" && sha256sum "$RUN.zip") > "$SHIP/$RUN.zip.sha256"
split -b "$PART" -d -a 3 "$Z" "$SHIP/$RUN.zip.part"

# 3. Orphan branch, pushed a batch at a time.
cd "$SHIP"
git init -q
parts=( "$RUN".zip.part* )
n=${#parts[@]}
for (( i = 0; i < n; i += BATCH )); do
    batch=( "${parts[@]:i:BATCH}" )
    last=$(( i + ${#batch[@]} ))
    git add "$RUN.zip.sha256" "${batch[@]}"
    git -c user.name="$NAME" -c user.email="$EMAIL" \
        commit -qm "obs data $RUN: parts $((i + 1))-$last of $n"
    git push -q "$REMOTE" "+HEAD:refs/heads/obs-data/$RUN"
    rm -f "${batch[@]}"
    echo "pushed parts $((i + 1))-$last of $n"
done
cd /
rm -rf "$SHIP"
echo "DONE: branch obs-data/$RUN ($n parts + $RUN.zip.sha256)"
echo "delete it once fetched: git push origin --delete obs-data/$RUN"
