# INSiNav ablations

All variants compose on top of `config/habitat_eval_insinav.yaml`. The launcher
isolates CUDA visibility, ROS master/topics, detector/SAM ports, ROS logs, and evaluation
output for each worker.

```bash
conda activate apexnav
./scripts/ablation/run_insinav_ablation.sh <variant> <gpu-id> <worker-id>
```

Core variants:

| Variant | Controlled change |
|---|---|
| `b0_apexnav_iin` | Crop LightGlue + sensor RGB-D stop; no active inspection, semantic gate, R2/R3, or MASt3R |
| `full` | R1 + R2 + R3, active inspection, crop ownership, fixed SE(2) geometry terminal |
| `wo_active_inspection` | Disable planned object observation viewpoints |
| `wo_multi_route` | Keep R3 only; disable both LightGlue routes |
| `wo_crop_ownership` | R1 full-frame evidence selects the DINO candidate without crop ownership matching |
| `wo_geometry_terminal` | Keep route triggers and MASt3R quality gate; replace fixed SE(2)/yaw with RGB-D stop/approach |

Run the six core rows with one isolated worker per list entry. Repeat a physical
GPU ID to place multiple workers on that GPU; each list position still receives a
unique worker ID and isolated ROS/VLM ports:

```bash
# Two L40 GPUs, three workers per GPU: all six core rows run concurrently.
./scripts/ablation/run_core_suite.sh 0 1 0 1 0 1

# One L40 GPU, three concurrent workers: each worker runs a second row afterward.
./scripts/ablation/run_core_suite.sh 0 0 0
```

Supplementary variants are `wo_negative_memory`, `wo_r1_boxed`,
`wo_r2_no_box`, `wo_r3_dino_direct`, and `wo_boxed_full_frame_gate`.

Use Hydra overrides after the first three launcher arguments for a smoke test:

```bash
./scripts/ablation/run_insinav_ablation.sh full 0 0 \
  test_epi_num=0 video_output_path='videos/ablation_smoke/full_{split}'
```

Use the same first 200 episodes for screening runs (this resumes up to 200):

```bash
./scripts/ablation/run_insinav_ablation.sh wo_r2_no_box 0 0 max_eval_episodes=200
```

Summarize route use after a run:

```bash
python scripts/ablation/summarize_routes.py \
  videos/ablation/full_val/route_metrics.jsonl
```
