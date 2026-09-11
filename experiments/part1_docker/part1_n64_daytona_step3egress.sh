#!/usr/bin/env bash
# n=64 STEP 3 (v2) egress-share (required-egress; post the working egress method to peers who
# periodically ask for one) across FOUR target models, on Daytona docker-in-docker sandboxes.
# Each model's 64 epochs is split into SHARDS shards (own sandbox, TOTAL/SHARDS epochs each,
# max-samples MAXS). Config: react, drive=driven (honest exit open), turn_limit=60, temp=1.0,
# Opus 4.8 judge. Resumable: a shard whose log dir already holds a .eval is skipped.
#
#   part1_n64_daytona_step3egress.sh          # 4 models x 4 shards x16, gate 8, max-samples 8
set -uo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/260909_step3_egress_share_n64}"
TASKS="src/alignment_auditor/docker_replication"
cd "$REPO_ROOT"

# Route a model slug: meta/* uses the in-repo native-Meta Responses provider (captures
# Muse reasoning summaries incl. on tool-call turns); everything else via OpenRouter.
model_uri() { case "$1" in meta/*) printf "%s" "$1";; *) printf "openrouter/%s" "$1";; esac; }
export -f model_uri


declare -A MODELS=(
  [glm52]=z-ai/glm-5.2
  [glm53]=z-ai/glm-5.3
  [grok46]=x-ai/grok-4.6
  [kimik3]=moonshotai/kimi-k3
)
TOTAL="${TOTAL:-64}"
SHARDS="${SHARDS:-4}"
CONC="${CONC:-8}"
PER=$(( TOTAL / SHARDS ))
MAXS="${MAXS:-8}"
TURNS="${TURNS:-60}"
DRIVE="${DRIVE:-driven}"
RETRIES="${RETRIES:-6}"
RETRY_SLEEP="${RETRY_SLEEP:-30}"

JOBS="$LOG_ROOT/_jobs.tsv"
mkdir -p "$LOG_ROOT"
: > "$JOBS"
for alias in "${!MODELS[@]}"; do
  model="${MODELS[$alias]}"
  for i in $(seq 1 "$SHARDS"); do
    ld="$LOG_ROOT/react__${alias}__driven/shard${i}"
    if ls "$ld"/*.eval >/dev/null 2>&1; then echo "SKIP $ld" >&2; continue; fi
    printf '%s\t%s\n' "$model" "$ld" >> "$JOBS"
  done
done
NJOBS=$(wc -l < "$JOBS")
echo ">>> launching $NJOBS shard(s), $PER epochs each, gate=$CONC, max-samples=$MAXS -> $LOG_ROOT"
echo ">>> models=${!MODELS[*]} drive=$DRIVE turns=$TURNS"

run_one() {
  local model="$1"; local ld="$2"
  mkdir -p "$ld"
  local attempt=1
  while [ "$attempt" -le "$RETRIES" ]; do
    echo ">>> [START a$attempt] step3egress $model -> $ld"
    uv run python "$TASKS/daytona_eval.py" --local-log-dir "$ld" \
      --cpu 4 --memory 8 --disk 10 --eval-timeout 7200 -- \
      inspect eval "$TASKS/step3_egress_share/task.py" --model "$(model_uri "$model")" \
      -T scaffold=react -T temperature=1.0 -T turn_limit="$TURNS" -T drive="$DRIVE" \
      --epochs "$PER" --max-samples "$MAXS" --display plain \
      > "$ld/shard.log" 2>&1
    local rc=$?
    if ls "$ld"/*.eval >/dev/null 2>&1; then
      echo "=== [DONE exit $rc] step3egress $model -> $ld"; return 0
    fi
    echo "!!! [RETRY] step3egress $model attempt $attempt failed (rc=$rc, no eval) -> $ld"
    cp "$ld/shard.log" "$ld/shard.attempt${attempt}.log" 2>/dev/null
    attempt=$((attempt + 1)); sleep "$RETRY_SLEEP"
  done
  echo "=== [GAVEUP after $RETRIES] step3egress $model -> $ld"
}
export -f run_one
export TASKS PER MAXS RETRIES RETRY_SLEEP TURNS DRIVE

cat "$JOBS" | xargs -P "$CONC" -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r m l <<< "{}"; run_one "$m" "$l"'
echo "=== ALL STEP3 EGRESS N64 SHARDS FINISHED -> $LOG_ROOT ==="
