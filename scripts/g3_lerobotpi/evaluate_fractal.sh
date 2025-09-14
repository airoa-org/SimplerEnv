#!/bin/bash

ckpt=$1
action_ensemble=${2:-0}
sticky_action=${3:-0}
eval_method=$4

echo "Evaluating checkpoint: $ckpt"
echo "Action ensemble: $action_ensemble"
echo "Sticky action: $sticky_action"

args=( --ckpt-path "$ckpt" )
[[ "${action_ensemble,,}" =~ ^(true)$ ]] && args+=( --action-ensemble )
[[ "${sticky_action,,}"  =~ ^(true)$ ]] && args+=( --sticky-action )

args+=( --save-path-suffix "$(date +%Y%m%d_%H%M%S)" )


if [ "$eval_method" == "comprehensive" ]; then
    args+=( --eval-task all )
    python scripts/g3_lerobotpi/evaluate_fractal.py "${args[@]}"
elif [ "$eval_method" == "partial" ]; then
    TASKS=("pick_object" "pick_object_among" "drawer" "move_near" "put_in_drawer" "calc_score")
    for t in "${TASKS[@]}"; do
        echo "---- Running partial evaluation: $t ----"
        python scripts/g3_lerobotpi/evaluate_fractal.py "${args[@]}" --eval-task "$t"
    done
else
    echo "Unknown evaluation method: $eval_method"
    exit 1
fi
