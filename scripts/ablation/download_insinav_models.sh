#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
torch_home="$repo_root/data/model_cache/torch"
hf_home="$repo_root/data/model_cache/huggingface"
dino_repo="$repo_root/data/models/dino"
checkpoint_dir="$torch_home/hub/checkpoints"
mast3r_checkpoint_dir="$repo_root/data/models/mast3r"

mkdir -p "$checkpoint_dir" "$hf_home" "$repo_root/data/models" "$mast3r_checkpoint_dir"

export TORCH_HOME="$torch_home"
export HF_HOME="$hf_home"
export HUGGINGFACE_HUB_CACHE="$hf_home/hub"

download_file() {
  local url="$1"
  local destination="$2"
  local minimum_bytes="$3"
  local current_bytes=0
  local partial_path="${destination}.part"
  local attempt=0

  if [[ -f "$destination" ]]; then
    current_bytes="$(stat -c '%s' "$destination")"
    if (( current_bytes >= minimum_bytes )); then
      echo "[skip] $destination ($current_bytes bytes)"
      return
    fi
    echo "[download] Existing file is too small; replacing $destination" >&2
    mv "$destination" "$partial_path"
  fi

  echo "[download] $url"
  echo "           -> $destination"
  for attempt in $(seq 1 20); do
    if curl --fail --location --connect-timeout 30 \
      --retry 5 --retry-delay 5 --continue-at - \
      --output "$partial_path" "$url"; then
      break
    fi
    if (( attempt == 20 )); then
      echo "Download failed after $attempt attempts: $url" >&2
      exit 1
    fi
    echo "[retry] Transfer failed; keeping the partial file and retrying in 5s ($attempt/20)" >&2
    sleep 5
  done

  current_bytes="$(stat -c '%s' "$partial_path")"
  if (( current_bytes < minimum_bytes )); then
    echo "Downloaded file is unexpectedly small: $partial_path ($current_bytes bytes)" >&2
    exit 1
  fi
  mv "$partial_path" "$destination"
}

adopt_cached_file() {
  local legacy_path="$1"
  local destination="$2"
  local minimum_bytes="$3"
  local legacy_bytes=0

  if [[ -f "$destination" || ! -f "$legacy_path" ]]; then
    return
  fi
  legacy_bytes="$(stat -c '%s' "$legacy_path")"
  if (( legacy_bytes < minimum_bytes )); then
    return
  fi
  echo "[migrate] $legacy_path -> $destination"
  cp --reflink=auto "$legacy_path" "${destination}.part"
  mv "${destination}.part" "$destination"
}

legacy_torch_home="${XDG_CACHE_HOME:-${HOME}/.cache}/torch"
legacy_hf_hub="${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface/hub"

# Reuse downloads from earlier runs before contacting the network.
adopt_cached_file "$legacy_torch_home/hub/checkpoints/dino_deitsmall16_pretrain.pth" \
  "$checkpoint_dir/dino_deitsmall16_pretrain.pth" 80000000
adopt_cached_file "$legacy_torch_home/hub/checkpoints/depth-save.pth" \
  "$checkpoint_dir/depth-save.pth" 4000000
adopt_cached_file "$legacy_torch_home/hub/checkpoints/disk_lightglue_v0-1_arxiv.pth" \
  "$checkpoint_dir/disk_lightglue_v0-1_arxiv.pth" 40000000

if [[ ! -e "$HUGGINGFACE_HUB_CACHE/models--bert-base-uncased" \
      && -d "$legacy_hf_hub/models--bert-base-uncased" ]]; then
  echo "[migrate] Existing BERT cache -> $HUGGINGFACE_HUB_CACHE"
  mkdir -p "$HUGGINGFACE_HUB_CACHE"
  cp -a "$legacy_hf_hub/models--bert-base-uncased" "$HUGGINGFACE_HUB_CACHE/"
fi

if [[ ! -f "$dino_repo/hubconf.py" ]]; then
  legacy_dino_repo="$legacy_torch_home/hub/facebookresearch_dino_main"
  if [[ -e "$dino_repo" ]]; then
    echo "DINO target exists but is not a valid checkout: $dino_repo" >&2
    exit 1
  elif [[ -d "$legacy_dino_repo" && -f "$legacy_dino_repo/hubconf.py" ]]; then
    echo "[migrate] Existing DINO checkout -> $dino_repo"
    cp -a "$legacy_dino_repo" "$dino_repo"
  else
    echo "[clone] facebookresearch/dino -> $dino_repo"
    git clone --depth 1 https://github.com/facebookresearch/dino.git "$dino_repo"
  fi
else
  echo "[skip] $dino_repo"
fi

# Visual backbone and instance matcher caches used by torch.hub/LightGlue/Kornia.
download_file \
  "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth" \
  "$checkpoint_dir/dino_deitsmall16_pretrain.pth" 80000000
download_file \
  "https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth" \
  "$checkpoint_dir/depth-save.pth" 4000000
download_file \
  "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/disk_lightglue.pth" \
  "$checkpoint_dir/disk_lightglue_v0-1_arxiv.pth" 40000000

# Detector and segmentor servers.
download_file \
  "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
  "$repo_root/data/groundingdino_swint_ogc.pth" 600000000
download_file \
  "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt" \
  "$repo_root/data/mobile_sam.pt" 35000000
download_file \
  "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt" \
  "$repo_root/data/yolov7-e6e.pt" 250000000

# Geometry terminal model. Use the official Hugging Face release because the
# Naver Labs checkpoint host is unreachable from some cloud providers.
download_file \
  "https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/resolve/main/config.json?download=true" \
  "$mast3r_checkpoint_dir/config.json" 500
download_file \
  "https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/resolve/main/model.safetensors?download=true" \
  "$mast3r_checkpoint_dir/model.safetensors" 2700000000

echo "[download] GroundingDINO BERT text encoder -> $HUGGINGFACE_HUB_CACHE"
"${PYTHON_BIN:-python}" - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    # GroundingDINO passes this exact identifier to Transformers at runtime, so
    # populate the matching cache namespace instead of its canonical redirect.
    repo_id="bert-base-uncased",
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ],
)
print(f"[download] BERT snapshot ready: {path}")
PY

verify_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | cut -d ' ' -f 1)"
  if [[ "$actual" != "$expected" ]]; then
    echo "Checksum mismatch for $path" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    exit 1
  fi
}

verify_sha256 "1566d50496f27f52f07fea6094fa29b2fdd6fae89da65bdd3ebc3b24ef6b7eb7" \
  "$checkpoint_dir/dino_deitsmall16_pretrain.pth"
verify_sha256 "9c2ee4ded238892dfa51569941372601e35e4a74aa6f84ea80053d2ab1c07abe" \
  "$checkpoint_dir/depth-save.pth"
verify_sha256 "b5b21d47ea24f2c5e501aec9c91b9716e4c8c3429a4dc1e615c133c4c9378335" \
  "$checkpoint_dir/disk_lightglue_v0-1_arxiv.pth"
verify_sha256 "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799" \
  "$repo_root/data/groundingdino_swint_ogc.pth"
verify_sha256 "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f" \
  "$repo_root/data/mobile_sam.pt"
verify_sha256 "b370120a414bf32b5d65fc808e5a32c8d9b3c63902d1bc41894fc9d86eccf9cb" \
  "$repo_root/data/yolov7-e6e.pt"
verify_sha256 "718eb93dc4f9e4332b60cc0041af962d712cbd346d7770ce35c5b22cff68eae4" \
  "$mast3r_checkpoint_dir/config.json"
verify_sha256 "0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb" \
  "$mast3r_checkpoint_dir/model.safetensors"

echo
echo "[verify] Downloaded model files:"
sha256sum \
  "$checkpoint_dir/dino_deitsmall16_pretrain.pth" \
  "$checkpoint_dir/depth-save.pth" \
  "$checkpoint_dir/disk_lightglue_v0-1_arxiv.pth" \
  "$repo_root/data/groundingdino_swint_ogc.pth" \
  "$repo_root/data/mobile_sam.pt" \
  "$repo_root/data/yolov7-e6e.pt" \
  "$mast3r_checkpoint_dir/config.json" \
  "$mast3r_checkpoint_dir/model.safetensors"
echo "[done] INSiNav model bundle is ready under $repo_root/data"
