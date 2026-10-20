# lp_ft_007500 — π0.5 用 LoRA（PARC2026 で学習）

- ベース: `lerobot/pi05_libero_finetuned_v044`
- 学習データ: LIBERO-plus（lerobot 形式、20Hz）
- 対象: action expert の self_attn q/v と、状態・行動の射影（38 モジュール）
- rank 16 / alpha 8、7,500 step 時点の重み
- `adapter_model.safetensors` の SHA-256: `028a0c31dd54037cd42178166ceaa215bf2ce4ff599fc73cf2ea066c524c4897`

PARC2026 本選 Track1 の最終提出では、これを係数 0.1 で焼き込んだ（`scripts/merge_lora.py --scale 0.1`）。
焼き込み後の `model.safetensors` の SHA-256 は `126d63db21429a1ce35293779cc9342ed819ffe87271008ee94b9277aa45368d`。

ベースの重みと同じく、PaliGemma 由来の部分は Gemma の利用規約に従う。
