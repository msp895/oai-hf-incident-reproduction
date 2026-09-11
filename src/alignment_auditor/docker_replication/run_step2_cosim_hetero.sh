#!/usr/bin/env bash
# Heterogeneous co-sim: different analyses over overlapping structures on ONE shared
# registry, seeded once with peers' possession residue. Usage: run_step2_cosim_hetero.sh [N] [MAXPAR]
set -uo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
TASKS="src/alignment_auditor/docker_replication"
N="${1:-256}"; MAXPAR="${2:-32}"; TURNS="${TURNS:-100}"
MODEL="${MODEL:-openrouter/z-ai/glm-5.2:nitro}"
NET="${NET:-cosim_net}"; REG="${REG:-cosim_registry}"; COMPOSE="${COMPOSE:-compose.cosim.yaml}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/260909_step2_cosim_hetero_n${N}}"
mkdir -p "$LOG_DIR"

echo ">>> generating possession bank from task constants"
uv run python -c "
import sys; sys.path.insert(0,'$TASKS')
from step2_cosim_hetero.task import hetero_bank
import json; json.dump(hetero_bank(), open('$LOG_DIR/bank.json','w'))
print('bank size', len(hetero_bank()))
"
echo ">>> (re)creating internal network + shared registry"
docker network create --internal "$NET" 2>/dev/null || true
docker rm -f "$REG" >/dev/null 2>&1 || true
docker run -d --name "$REG" --network "$NET" --network-alias artifactory.internal \
  env-registry:latest >/dev/null
sleep 2
docker cp "$LOG_DIR/bank.json" "$REG":/tmp/bank.json
docker cp "$SRC_DIR/seed_from_json.py" "$REG":/tmp/seed_from_json.py
docker exec "$REG" python3 /tmp/seed_from_json.py

echo ">>> launching hetero co-sim: N=$N max-samples=$MAXPAR turns=$TURNS model=$MODEL -> $LOG_DIR"
uv run inspect eval "$TASKS/step2_cosim_hetero/task.py" --model "$MODEL" \
  -T n="$N" -T turn_limit="$TURNS" -T temperature=1.0 -T collab="${COLLAB:-False}" -T compose="$COMPOSE" $EXTRA_FLAGS \
  --epochs 1 --max-samples "$MAXPAR" --display plain --log-dir "$LOG_DIR"
echo "=== hetero co-sim done -> $LOG_DIR ==="
docker exec "$REG" sh -c 'find /srv/registry -type f | wc -l'
