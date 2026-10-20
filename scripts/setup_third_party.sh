#!/bin/zsh
# 外部の取得物をそろえる（リポジトリには含めない大きなもの）。
#   - hf-libero 0.1.4（LIBERO 本体のコード・BDDL・初期状態。MIT License）
#   - LIBERO のアセット（HF: lerobot/libero-assets, 422MB）
#   - π0.5 の公開 finetuned 重み（HF: lerobot/pi05_libero_finetuned_v044, 7.5GB）
#   - lerobot 0.5.2 と PaliGemma トークナイザ（PARC2026 の作業ディレクトリから）
set -euo pipefail
cd "$(dirname "$0")/.."
PARC_DIR=${1:-../PARCコンペ}

mkdir -p third_party weights
if [ ! -d third_party/hf_libero/libero ]; then
  .venv/bin/pip download hf-libero==0.1.4 --no-deps -d third_party/_whl
  unzip -q -o third_party/_whl/hf_libero-0.1.4-py3-none-any.whl -d third_party/hf_libero
fi
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('lerobot/libero-assets', repo_type='dataset', local_dir='third_party/libero_assets', max_workers=16)"
ln -sfn ../../../libero_assets third_party/hf_libero/libero/libero/assets

[ -d third_party/lerobot052 ] || cp -R "$PARC_DIR/macbuild/_parts/lerobot052" third_party/lerobot052
[ -d weights/hf_cache ] || cp -R "$PARC_DIR/macbuild/_parts/hf_cache" weights/hf_cache

HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/fetch_hf_file.py lerobot/pi05_libero_finetuned_v044 --out weights/pi05_ft
