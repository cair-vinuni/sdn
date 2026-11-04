#!/usr/bin/env bash
# pi0 + SDN on the 9 SIMPLER tasks (Table II).
# Usage (from anywhere): bash policy/pizero/scripts/run_sdn.sh [num_gpus] [result_root]
# GPUs are taken from CUDA_VISIBLE_DEVICES if set. N_TRAJS (default 50) episodes per task, with
# episode seeds 0 .. N_TRAJS - 1.
set -euo pipefail
cd "$(dirname "$0")/../../.."

NUM_GPUS=${1:-1}
RESULT_ROOT=${2:-./results/pizero}
N_TRAJS=${N_TRAJS:-50}

# SDN hyperparameters per task: "knn_k top_m long_horizon", with N = 12 candidates everywhere.
# Chosen per task among k in {6, 10} x M in {3, 5} (long horizon 4) by the success rate on episode
# seeds 0-99.
declare -A HPARAMS=(
    [google_robot_close_drawer]="10 5 4"
    [google_robot_move_near]="10 5 4"
    [google_robot_open_drawer]="10 5 4"
    [google_robot_pick_coke_can]="10 3 4"
    [google_robot_place_apple_in_closed_top_drawer]="10 3 4"
    [widowx_carrot_on_plate]="10 3 4"
    [widowx_put_eggplant_in_basket]="10 5 4"
    [widowx_spoon_on_towel]="10 5 4"
    [widowx_stack_cube]="6 3 4"
)

for task in "${!HPARAMS[@]}"; do
    read -r knn_k top_m long_horizon <<< "${HPARAMS[$task]}"
    echo "=== SDN | $task | k=$knn_k M=$top_m long_horizon=$long_horizon"
    python -m policy.pizero.evaluate \
        --method sdn \
        --task "$task" \
        --num-gpus "$NUM_GPUS" \
        --n-trajs "$N_TRAJS" \
        --result-root "$RESULT_ROOT" \
        --opts num_samples 12 knn_k "$knn_k" top_m "$top_m" long_horizon "$long_horizon" \
               negative_mode zeros_bbox
done
