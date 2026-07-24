#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <variant> <physical-gpu-id> <worker-id> [Hydra overrides ...]" >&2
  exit 2
fi

variant="$1"
gpu_id="$2"
worker_id="$3"
shift 3

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
variant_cfg="$repo_root/config/insinav_ablation/$variant.yaml"
if [[ ! -f "$variant_cfg" ]]; then
  echo "Unknown ablation '$variant'. Available variants:" >&2
  find "$repo_root/config/insinav_ablation" -maxdepth 1 -name '*.yaml' \
    -printf '  %f\n' | sed 's/\.yaml$//' | sort >&2
  exit 2
fi
if [[ ! "$worker_id" =~ ^[0-9]+$ ]]; then
  echo "worker-id must be a non-negative integer" >&2
  exit 2
fi
if (( worker_id > 5000 )); then
  echo "worker-id must be <= 5000 so derived TCP ports remain valid" >&2
  exit 2
fi

ros_port=$((11311 + worker_id))
port_offset=$((worker_id * 10))
dino_port=$((12181 + port_offset))
sam_port=$((12183 + port_offset))
yolo_port=$((12184 + port_offset))
run_dir="$repo_root/videos/ablation/${variant}_val"
mkdir -p "$run_dir/ros_home" "$run_dir/ros_logs" "$run_dir/matplotlib"
local_llm_answers="$run_dir/llm_answer_hm3d.txt"
local_llm_responses="$run_dir/llm_response_list.txt"
if [[ ! -f "$local_llm_answers" ]]; then
  cp "$repo_root/llm/answers/llm_answer_hm3d.txt" "$local_llm_answers"
fi
touch "$local_llm_responses"

export CUDA_VISIBLE_DEVICES="$gpu_id"
export ROS_MASTER_URI="http://127.0.0.1:$ros_port"
export ROS_HOSTNAME="127.0.0.1"
export ROS_HOME="$run_dir/ros_home"
export ROS_LOG_DIR="$run_dir/ros_logs"
export MPLCONFIGDIR="$run_dir/matplotlib"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

python_bin="${PYTHON_BIN:-python}"
if [[ -f "$repo_root/devel/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "$repo_root/devel/setup.bash"
fi

background_pids=()
cleanup() {
  local pid
  for pid in "${background_pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${background_pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "[ablation] variant=$variant gpu=$gpu_id worker=$worker_id"
echo "[ablation] ROS_MASTER_URI=$ROS_MASTER_URI VLM_PORTS=$dino_port,$sam_port,$yolo_port"
echo "[ablation] output=$run_dir"

roscore -p "$ros_port" >"$run_dir/roscore.log" 2>&1 &
background_pids+=("$!")

CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -m vlm.detector.grounding_dino \
  --port "$dino_port" >"$run_dir/grounding_dino.log" 2>&1 &
background_pids+=("$!")
CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -m vlm.segmentor.sam \
  --port "$sam_port" >"$run_dir/mobile_sam.log" 2>&1 &
background_pids+=("$!")
CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -m vlm.detector.yolov7 \
  --port "$yolo_port" >"$run_dir/yolov7.log" 2>&1 &
background_pids+=("$!")

ros_ready=0
for _ in $(seq 1 60); do
  if rosparam list >/dev/null 2>&1; then
    ros_ready=1
    break
  fi
  sleep 1
done
if [[ "$ros_ready" -ne 1 ]]; then
  echo "ROS master did not become ready; see $run_dir/roscore.log" >&2
  exit 1
fi

wait_for_vlm() {
  local port="$1"
  local endpoint="$2"
  local log_path="$3"
  local ready=0
  for _ in $(seq 1 240); do
    if curl --max-time 1 -s -o /dev/null "http://127.0.0.1:$port/$endpoint"; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "$endpoint server did not become ready; see $log_path" >&2
    return 1
  fi
}
wait_for_vlm "$dino_port" gdino "$run_dir/grounding_dino.log"
wait_for_vlm "$sam_port" mobile_sam "$run_dir/mobile_sam.log"
wait_for_vlm "$yolo_port" yolov7 "$run_dir/yolov7.log"

object_viewpoint_enabled=true
semantic_gate_enabled=true
negative_memory_args=()
case "$variant" in
  b0_apexnav_iin)
    object_viewpoint_enabled=false
    semantic_gate_enabled=false
    ;;
  wo_active_inspection)
    object_viewpoint_enabled=false
    ;;
  wo_negative_memory)
    negative_memory_args+=(
      "object_viewpoint_max_failed_inspections:=1000"
      "object_viewpoint_failure_cooldown_sequences:=0"
      "object_viewpoint_retry_score_margin:=0.0"
    )
    ;;
esac
planner_args=(
  "eval_config:=habitat_eval_insinav.yaml"
  "object_viewpoint_enabled:=$object_viewpoint_enabled"
  "semantic_gate_enabled:=$semantic_gate_enabled"
  "${negative_memory_args[@]}"
)

roslaunch exploration_manager exploration.launch "${planner_args[@]}" \
  >"$run_dir/planner.log" 2>&1 &
background_pids+=("$!")
sleep 3
if ! kill -0 "${background_pids[-1]}" 2>/dev/null; then
  echo "Planner exited during startup; see $run_dir/planner.log" >&2
  exit 1
fi

cd "$repo_root"
"$python_bin" habitat_evaluation.py --dataset insinav \
  "+insinav_ablation=$variant" \
  "detector.grounding_dino_server_port=$dino_port" \
  "detector.sam_server_port=$sam_port" \
  "detector.yolo_server_port=$yolo_port" \
  "llm.llm_answer_path=$local_llm_answers" \
  "llm.llm_response_path=$local_llm_responses" \
  "$@" 2>&1 | tee "$run_dir/evaluation.log"
