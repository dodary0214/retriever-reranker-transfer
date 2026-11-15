#!/usr/bin/env bash
# CPU-only analysis from stored model outputs (no GPU, no model downloads).
# Inputs:  $RRT_DATA (Experiment 1), $RRT_CONTROLLED_DATA (Experiment 2); defaults: data/cross_retriever, data/controlled_context
# Outputs: $RRT_OUT (default: outputs/) -> controlled_context/, tables/, figures/, derived/, RESULTS.md, logs/
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export RRT_OUT=${RRT_OUT:-$REPO/outputs}
export RRT_CONTROLLED_DATA=${RRT_CONTROLLED_DATA:-$REPO/data/controlled_context}
export RRT_CONTROLLED_RESULTS=${RRT_CONTROLLED_RESULTS:-$RRT_OUT/controlled_context}
export SEED=${SEED:-0}
CTRL=$RRT_CONTROLLED_DATA/controlled
mkdir -p "$RRT_OUT/logs" "$RRT_CONTROLLED_RESULTS"

step() {  # step <name> <script> [args...]
  local name=$1; shift
  echo "$(date '+%F %T') $name"
  python -W ignore "$@" > "$RRT_OUT/logs/$name.log" 2>&1 || { echo "FAILED: $name (see $RRT_OUT/logs/$name.log)"; exit 1; }
}

# Experiment 2: per-unit reversal tables (C5 = first three permutations) and the model comparison (m1_*.csv)
step exp2_zephyr experiments/controlled_context/analyze.py --rank "$CTRL/rank_out.jsonl" --outdir "$RRT_CONTROLLED_RESULTS" \
  --tag zephyr --c5-perms 3
step exp2_first experiments/controlled_context/analyze.py --rank "$CTRL/models/rank_out_first.jsonl" \
  --outdir "$RRT_CONTROLLED_RESULTS" --tag first --model-name FIRST --c5-perms 3
step exp2_compare experiments/controlled_context/compare_models.py "$RRT_CONTROLLED_RESULTS"

# Experiment 1: main tables and figure values, then the analyses over all 15 four-reranker sets
step analyze analysis/analyze.py
step appendix_qwen analysis/robustness/appendix_qwen.py
step ksets analysis/query_agreement/ksets.py
step fig3_spearman analysis/query_agreement/fig3_spearman.py
step ksets_robust analysis/robustness/ksets_robust.py
step ksets_selection_ndcg analysis/evaluation_transfer/ksets_selection_ndcg.py
step appendixB_extra analysis/robustness/appendixB_extra.py
step diag_rerun_windows analysis/pair_reversals/diag_rerun_windows.py
step matched_swap analysis/query_agreement/matched_swap.py
echo "$(date '+%F %T') ANALYSIS_DONE"
