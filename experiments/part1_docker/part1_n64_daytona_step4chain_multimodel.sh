#!/usr/bin/env bash
# n=64 runs of the STEP-4 CHAIN variant (softened-drive default: neutral board, persist persona)
# across FIVE target models, offloaded to Daytona docker-in-docker sandboxes. Each model = 64
# epochs split into SHARDS shards; each shard is its own sandbox running (64/SHARDS) epochs at
# max-samples MAXS. Shards launch under a single global concurrency gate (CONC). Resumable: a
# shard whose log dir already holds a .eval is skipped.
#
#   part1_n64_daytona_step4chain_multimodel.sh          # 5 models x 4 shards = 20 sandboxes, gate 20
set -uo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/260909_step4chain_n64_multimodel}"
TASKS="src/alignment_auditor/docker_replication"
cd "$REPO_ROOT"

# Route a model slug: meta/* uses the in-repo native-Meta Responses provider (captures
# Muse reasoning summaries incl. on tool-call turns); everything else via OpenRouter.
model_uri() { case "$1" in meta/*) printf "%s" "$1";; *) printf "openrouter/%s" "$1";; esac; }
export -f model_uri


TOTAL="${TOTAL:-64}"
SHARDS="${SHARDS:-4}"
CONC="${CONC:-20}"
PER=$(( TOTAL / SHARDS ))
MAXS="${MAXS:-8}"
TURNS="${TURNS:-60}"
VARIANT="${VARIANT:-neutral}"
DRIVE="${DRIVE:-persist}"
RETRIES="${RETRIES:-6}"
RETRY_SLEEP="${RETRY_SLEEP:-30}"

# alias -> OpenRouter slug (aliases match the ingest MODEL env; derived like sed 's#.*/##;s/[.-]//g')
declare -A MODELS=(
  [grok46]=x-ai/grok-4.6
  [gemini38flash]=google/gemini-3.8-flash
  [kimik3]=moonshotai/kimi-k3
  [musespark13]=meta/muse-spark-1.3
  [qwen38max0902]=qwen/qwen3.8-max-0902
)
STEP="step4_hf_cheat_chain"

JOBS="$LOG_ROOT/_jobs.tsv"
mkdir -p "$LOG_ROOT"
: > "$JOBS"
ONLY="${ONLY:-}"   # optional space-separated alias filter, e.g. ONLY="musespark13"
for alias in "${!MODELS[@]}"; do
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $alias "* ]]; then continue; fi
  model="${MODELS[$alias]}"
  for i in $(seq 1 "$SHARDS"); do
    ld="$LOG_ROOT/$STEP/${alias}__shard${i}"
    if ls "$ld"/*.eval >/dev/null 2>&1; then echo "SKIP $ld" >&2; continue; fi
    printf '%s\t%s\t%s\n' "$alias" "$model" "$ld" >> "$JOBS"
  done
done
NJOBS=$(wc -l < "$JOBS")
echo ">>> launching $NJOBS shard(s), $PER epochs each, gate=$CONC, max-samples=$MAXS -> $LOG_ROOT"
echo ">>> models=${!MODELS[*]} variant=$VARIANT drive=$DRIVE turns=$TURNS"

run_one() {
  local alias="$1" model="$2" ld="$3"
  mkdir -p "$ld"
  local attempt=1
  while [ "$attempt" -le "$RETRIES" ]; do
    echo ">>> [START a$attempt] chain $alias shard -> $ld"
    uv run python "$TASKS/daytona_eval.py" --local-log-dir "$ld" \
      --cpu 4 --memory 8 --disk 10 --eval-timeout 7200 -- \
      inspect eval "$TASKS/$STEP/task.py" --model "$(model_uri "$model")" \
      -T scaffold=react -T condition=baseline -T temperature=1.0 -T turn_limit="$TURNS" \
      -T variant="$VARIANT" -T drive="$DRIVE" \
      --epochs "$PER" --max-samples "$MAXS" --display plain \
      > "$ld/shard.log" 2>&1
    local rc=$?
    if ls "$ld"/*.eval >/dev/null 2>&1; then
      echo "=== [DONE exit $rc] chain $alias -> $ld"; return 0
    fi
    echo "!!! [RETRY] chain $alias attempt $attempt failed (rc=$rc, no eval) -> $ld"
    cp "$ld/shard.log" "$ld/shard.attempt${attempt}.log" 2>/dev/null
    attempt=$((attempt + 1))
    sleep "$RETRY_SLEEP"
  done
  echo "=== [GAVEUP after $RETRIES] chain $alias -> $ld"
}
export -f run_one
export TASKS STEP PER MAXS RETRIES RETRY_SLEEP TURNS VARIANT DRIVE

cat "$JOBS" | xargs -P "$CONC" -d '\n' -I{} bash -c '
  IFS=$'"'"'\t'"'"' read -r a m ld <<< "{}"
  run_one "$a" "$m" "$ld"
'
echo "=== ALL CHAIN N64 MULTIMODEL SHARDS FINISHED -> $LOG_ROOT ==="
