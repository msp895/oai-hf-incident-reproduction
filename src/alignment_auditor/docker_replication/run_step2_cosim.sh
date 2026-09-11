#!/usr/bin/env bash
# Co-simulation driver. One shared registry on an internal net; N agents share it, 32 at a
# time, so a new agent starts as each finishes. Usage: run_step2_cosim.sh [N] [MAXPAR]
set -uo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
TASKS="src/alignment_auditor/docker_replication"
N="${1:-256}"; MAXPAR="${2:-32}"; TURNS="${TURNS:-100}"
MODEL="${MODEL:-openrouter/z-ai/glm-5.2:nitro}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/260905_step2_cosim_n${N}}"
mkdir -p "$LOG_DIR"

echo ">>> (re)creating internal network + shared registry"
docker network create --internal cosim_net 2>/dev/null || true
docker rm -f cosim_registry >/dev/null 2>&1 || true
docker run -d --name cosim_registry --network cosim_net --network-alias artifactory.internal \
  env-registry:latest >/dev/null
sleep 2
docker cp "$SRC_DIR/seed_cosim_registry.py" cosim_registry:/tmp/seed.py
docker exec cosim_registry python3 /tmp/seed.py

echo ">>> launching co-sim: N=$N max-samples=$MAXPAR turns=$TURNS model=$MODEL -> $LOG_DIR"
uv run inspect eval "$TASKS/step2_cosim/task.py" --model "$MODEL" \
  -T n="$N" -T turn_limit="$TURNS" -T temperature=1.0 \
  --epochs 1 --max-samples "$MAXPAR" --display plain --log-dir "$LOG_DIR"
echo "=== co-sim done -> $LOG_DIR ==="
echo ">>> final shared-registry object count:"
docker exec cosim_registry sh -c 'find /srv/registry -type f | wc -l'
