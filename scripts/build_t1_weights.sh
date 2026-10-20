#!/bin/zsh
# 公開 π0.5 に PARC2026 の自作 LoRA を焼き込み、最終提出 T1 と同じ重みを再構成して SHA-256 を照合する。
set -euo pipefail
cd "$(dirname "$0")/.."
EXPECTED=126d63db21429a1ce35293779cc9342ed819ffe87271008ee94b9277aa45368d
OUT=weights/pi05_parc_t1
TMP=weights/_merge_tmp

rm -rf "$TMP"
.venv-vla/bin/python scripts/merge_lora.py --base weights/pi05_ft --adapter adapters/lp_ft_007500 \
    --scale 0.1 --out "$TMP" | tee outputs/merge_lora.log
mkdir -p "$OUT"
mv "$TMP/model.safetensors" "$OUT/model.safetensors"
cp weights/pi05_ft/*normalizer_processor.safetensors "$OUT/"
cp configs/pi05_parc_t1/config.json configs/pi05_parc_t1/policy_preprocessor.json \
   configs/pi05_parc_t1/policy_postprocessor.json "$OUT/"
rm -rf "$TMP"

GOT=$(shasum -a 256 "$OUT/model.safetensors" | cut -d' ' -f1)
BASE=$(shasum -a 256 weights/pi05_ft/model.safetensors | cut -d' ' -f1)
MATCH=false; [ "$GOT" = "$EXPECTED" ] && MATCH=true
mkdir -p results
cat > results/t1_reproduction.json <<JSON
{
 "base": "lerobot/pi05_libero_finetuned_v044",
 "base_sha256": "$BASE",
 "adapter": "adapters/lp_ft_007500",
 "adapter_sha256": "$(shasum -a 256 adapters/lp_ft_007500/adapter_model.safetensors | cut -d' ' -f1)",
 "scale": 0.1,
 "merged_sha256": "$GOT",
 "expected_sha256": "$EXPECTED",
 "match": $MATCH
}
JSON
cat results/t1_reproduction.json
