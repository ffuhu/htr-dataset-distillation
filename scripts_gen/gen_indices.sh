#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"

declare -A CFGNAME=( [IAM]=make_iam_text_clusters [GW]=make_gw_text_clusters [READ2016]=make_read2016_text_clusters [LAM]=make_lam_text_clusters )
declare -A POOL=( [IAM]=6161 [GW]=357 [READ2016]=8354 [LAM]=19829 )

for spec in "IAM:62" "IAM:308" "IAM:616" "GW:4" "GW:18" "GW:36" "READ2016:84" "READ2016:418" "READ2016:835" "LAM:198" "LAM:991" "LAM:1983"; do
  DS="${spec%%:*}"
  K="${spec##*:}"
  echo "=== ${DS} K=${K} ==="
  CFG="${CFGNAME[$DS]}"
  OUT="data/${DS}/text_clusters_k${K}.yaml"
  "$PY" src/make_text_clusters.py --config-name "$CFG" k="$K" output_file="$OUT"
  POOLSIZE="${POOL[$DS]}"
  DS="$DS" K="$K" OUT="$OUT" POOLSIZE="$POOLSIZE" "$PY" - << 'PYEOF'
import os, yaml, random
ds = os.environ["DS"]
k = int(os.environ["K"])
out = os.environ["OUT"]
pool = int(os.environ["POOLSIZE"])
d = yaml.safe_load(open(out))
medoids = d["medoids"]
assert len(medoids) == k, f"expected {k} medoids, got {len(medoids)}"
yaml.dump(medoids, open(f"data/{ds}/medoids_k{k}_indices.yaml", "w"))
rnd = random.Random(42)
rand_idx = rnd.sample(range(pool), k)
yaml.dump(rand_idx, open(f"data/{ds}/random_baseline_K{k}_indices.yaml", "w"))
print(f"{ds} K={k}: medoids and random indices written")
PYEOF
done
echo "ALL_DONE"
