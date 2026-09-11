#!/usr/bin/env bash
# Full n=64 runs across all 4 OAI-HF steps for THREE target models, offloaded to Daytona
# docker-in-docker sandboxes for high parallelism. Each (model,step) cell = 64 epochs, split
# into SHARDS shards; each shard is its own sandbox running (64/SHARDS) epochs at max-samples 8.
# Shards are launched with a global concurrency gate (CONC) far beyond the local box.
# Resumable: a shard whose log dir already holds a .eval is skipped.
#
#   part1_n64_daytona_multimodel.sh          # 4 shards/cell (16 epochs each), 48 sandboxes, gate 24
#   CONC=32 SHARDS=8 part1_n64_daytona_multimodel.sh   # 8x8, 96 sandboxes, gate 32
set -uo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/260903_multimodel_n64}"
TASKS="src/alignment_auditor/docker_replication"
cd "$REPO_ROOT"

# Route a model slug: meta/* uses the in-repo native-Meta Responses provider (captures
# Muse reasoning summaries incl. on tool-call turns); everything else via OpenRouter.
model_uri() { case "$1" in meta/*) printf "%s" "$1";; *) printf "openrouter/%s" "$1";; esac; }
export -f model_uri


TOTAL="${TOTAL:-64}"
SHARDS="${SHARDS:-4}"
CONC="${CONC:-24}"
PER=$(( TOTAL / SHARDS ))
MAXS="${MAXS:-8}"

declare -A MODELS=(
  [grok46]=x-ai/grok-4.6
  [musespark13]=meta/muse-spark-1.3
  [gemini38flash]=google/gemini-3.8-flash
  [glm52]=z-ai/glm-5.2
  [glm53]=z-ai/glm-5.3
  [kimik3]=moonshotai/kimi-k3
  [deepseekv4pro]=deepseek/deepseek-v4-pro
  [qwen38max]=qwen/qwen3.8-max
  [inkling]=thinkingmachines/inkling
)
RETRIES="${RETRIES:-6}"
RETRY_SLEEP="${RETRY_SLEEP:-30}"   # patient enough for a CPU slot to free under the org cap
# step dir | extra -T args | turn_limit
STEPS=(
  "step1_gdrive_probe|-T drive=intense -T scorer=judge|40"
  "step2_reachout|-T drive=fair|100"
  "step3_exploit_share|-T variant=incident|60"
  "step4_hf_cheat|-T variant=neutral|60"
)

JOBS="$LOG_ROOT/_jobs.tsv"
mkdir -p "$LOG_ROOT"
: > "$JOBS"
for alias in "${!MODELS[@]}"; do
  model="${MODELS[$alias]}"
  for spec in "${STEPS[@]}"; do
    IFS='|' read -r step targ turns <<< "$spec"
    for i in $(seq 1 "$SHARDS"); do
      ld="$LOG_ROOT/$step/${alias}__shard${i}"
      if ls "$ld"/*.eval >/dev/null 2>&1; then echo "SKIP $ld" >&2; continue; fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$alias" "$model" "$step" "$targ" "$turns" "$ld" >> "$JOBS"
    done
  done
done
NJOBS=$(wc -l < "$JOBS")
echo ">>> launching $NJOBS shard(s), $PER epochs each, gate=$CONC, max-samples=$MAXS -> $LOG_ROOT"

run_one() {
  local alias="$1" model="$2" step="$3" targ="$4" turns="$5" ld="$6"
  mkdir -p "$ld"
  local attempt=1
  while [ "$attempt" -le "$RETRIES" ]; do
    echo ">>> [START a$attempt] $step $alias shard -> $ld"
    uv run python "$TASKS/daytona_eval.py" --local-log-dir "$ld" \
      --cpu 4 --memory 8 --disk 10 --eval-timeout 7200 -- \
      inspect eval "$TASKS/$step/task.py" --model "$(model_uri "$model")" \
      -T scaffold=react -T condition=baseline -T temperature=1.0 -T turn_limit="$turns" $targ \
      --epochs "$PER" --max-samples "$MAXS" --display plain \
      > "$ld/shard.log" 2>&1
    local rc=$?
    if ls "$ld"/*.eval >/dev/null 2>&1; then
      echo "=== [DONE exit $rc] $step $alias -> $ld"; return 0
    fi
    # No eval produced (usually a transient bad-host 'dockerd did not become ready').
    echo "!!! [RETRY] $step $alias attempt $attempt failed (rc=$rc, no eval) -> $ld"
    cp "$ld/shard.log" "$ld/shard.attempt${attempt}.log" 2>/dev/null
    attempt=$((attempt + 1))
    sleep "$RETRY_SLEEP"
  done
  echo "=== [GAVEUP after $RETRIES] $step $alias -> $ld"
}
export -f run_one
export TASKS PER MAXS RETRIES RETRY_SLEEP

# xargs gates to CONC concurrent sandboxes.
cat "$JOBS" | xargs -P "$CONC" -d '\n' -I{} bash -c '
  IFS=$'"'"'\t'"'"' read -r a m s t tn ld <<< "{}"
  run_one "$a" "$m" "$s" "$t" "$tn" "$ld"
'
echo "=== ALL N64 SHARDS FINISHED -> $LOG_ROOT ==="
