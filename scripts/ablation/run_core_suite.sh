#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <physical-gpu-id> [physical-gpu-id ...]" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gpus=("$@")
variants=(
  full
  b0_apexnav_iin
  wo_active_inspection
  wo_multi_route
  wo_crop_ownership
  wo_geometry_terminal
)

worker_pids=()
for slot in "${!gpus[@]}"; do
  (
    startup_delay=$((slot * 8))
    if (( startup_delay > 40 )); then
      startup_delay=40
    fi
    sleep "$startup_delay"
    for ((index=slot; index<${#variants[@]}; index+=${#gpus[@]})); do
      "$script_dir/run_insinav_ablation.sh" \
        "${variants[$index]}" "${gpus[$slot]}" "$slot"
    done
  ) &
  worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
