"""π0.5 推論（PARC2026 本選 Track1 最終提出と同じ推論経路）。

PARC2026 で最終提出した T1 モデル（公開 finetuned π0.5 ＋ 自作 LoRA を係数 0.1 で焼き込み、
n_action_steps=12）を、提出時と同じ手順で動かす:

  画像を 256 へ拡大 → LIBERO の観測前処理（180° 回転・state 8 次元化）→ 学習時の正規化
  → select_action（12 ステップずつ実行）→ 逆正規化

PARC で 0 点を出した落とし穴への対策もそのまま持ち込んでいる:
  - from_pretrained は読み込み失敗を print で握り潰す → 重みを値で照合してから起動する
  - 正規化統計が空でも素通しで動いてしまう → 統計の実在と非退化を確かめる
  - フローマッチングの初期ノイズ → /reset ごとに乱数を固定（同じ入力なら同じ軌道）

実行環境: Python 3.10+ / torch 2.8 / transformers 5.3.0 / lerobot 0.5.2（PARC で vendor したもの）。
Apple GPU (MPS) では float32 で動かす。
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np

IMG = 256
_WS = re.compile(r"\s+")


def normalize_instruction(text: str) -> str:
    """PARC で効いた最小の正規化（小文字化・空白・動詞形）。語を削る正規化はしない。"""
    t = _WS.sub(" ", str(text or "")).strip().lower()
    return t.replace("pick the ", "pick up the ")


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    import torch

    if img.shape[0] == size and img.shape[1] == size:
        return np.ascontiguousarray(img)
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8).contiguous().numpy()


def _verify_weights_loaded(policy, model_dir: Path, n_checked: int = 12) -> int:
    import torch
    from safetensors import safe_open

    params = dict(policy.named_parameters())
    checked = 0
    with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        keys = sorted(f.keys())
        for key in keys[:: max(1, len(keys) // n_checked)]:
            p = params.get(key if key.startswith("model.") else f"model.{key}")
            if p is None or p.ndim == 0:
                continue
            want = f.get_slice(key)[:1].to(torch.float32)
            got = p.detach()[:1].to("cpu", torch.float32)
            if got.shape != want.shape or not torch.equal(got, want):
                raise SystemExit(f"重みが読み込まれていない: {key}")
            checked += 1
    if checked == 0:
        raise SystemExit("model.safetensors と照合できた重みが無い")
    return checked


def _verify_normalization(model_dir: Path) -> None:
    import glob

    from safetensors import safe_open

    files = sorted(glob.glob(str(model_dir / "*normalizer_processor.safetensors")))
    if not files:
        raise SystemExit("正規化統計が無い（素通しになり、行動が壊れる）")
    seen = set()
    for p in files:
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                feat, _, stat = k.rpartition(".")
                if stat == "std" and float(f.get_tensor(k).abs().max()) > 0:
                    seen.add(feat)
    missing = {"action", "observation.state"} - seen
    if missing:
        raise SystemExit(f"正規化統計が退化している: {sorted(missing)}")


class Pi05T1Policy:
    def __init__(self, model_dir: str | Path, device: str = "mps", hf_home: str | Path | None = None) -> None:
        if hf_home:  # PaliGemma のトークナイザ（ゲート付き）はローカルのキャッシュから読む
            os.environ["HF_HOME"] = str(hf_home)
            os.environ["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.configs.types import FeatureType
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline

        self.torch = torch
        self.device = device
        model_dir = Path(model_dir)
        t0 = time.monotonic()
        cfg = PreTrainedConfig.from_pretrained(str(model_dir))
        cfg.device = device
        if device in ("mps", "cpu"):  # MPS では bf16 の行列積が落ちることがある（PARC で実測）ので fp32 で組む
            cfg.dtype = "float32"
        # 3.5B パラメータの乱数初期化（約 80 秒）は直後に重みで上書きされるので省く。
        # strict=True で読み、さらに値で照合するので安全（PARC T1 提出時と同じ手当て）
        try:
            from transformers.initialization import no_init_weights

            skip_init = no_init_weights()
        except Exception:  # noqa: BLE001
            from contextlib import nullcontext

            skip_init = nullcontext()
        with skip_init:
            policy = PI05Policy.from_pretrained(str(model_dir), config=cfg, strict=True)
        self.policy = policy.to(device).eval()
        self.load_s = time.monotonic() - t0
        self.n_verified = _verify_weights_loaded(self.policy, model_dir)
        _verify_normalization(model_dir)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(model_dir),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.env_pre = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
        visuals = [k for k, v in self.policy.config.input_features.items()
                   if v.type == FeatureType.VISUAL and "empty_camera" not in k]
        wrist = [k for k in visuals if k.endswith(("wrist", "image2")) or "eye_in_hand" in k]
        front = [k for k in visuals if k not in wrist]
        assert len(front) == 1 and len(wrist) == 1, visuals
        self.front_key = front[0].rsplit(".", 1)[-1]
        self.wrist_key = wrist[0].rsplit(".", 1)[-1]
        self.instruction = ""
        self.seed = 0
        self.n_action_steps = int(self.policy.config.n_action_steps)

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        self.policy.reset()
        self.instruction = normalize_instruction(instruction)
        self.seed = 0 if seed is None else int(seed)
        self.torch.manual_seed(self.seed)

    def get_action(self, obs: dict) -> np.ndarray:
        from lerobot.envs.utils import preprocess_observation

        gym = {
            "pixels": {
                self.front_key: _resize(obs["agentview_image"], IMG),
                self.wrist_key: _resize(obs["robot0_eye_in_hand_image"], IMG),
            },
            "robot_state": {
                "eef": {
                    "pos": np.asarray(obs["robot0_eef_pos"], np.float32).reshape(1, 3),
                    "quat": np.asarray(obs["robot0_eef_quat"], np.float32).reshape(1, 4),
                },
                "gripper": {"qpos": np.asarray(obs["robot0_gripper_qpos"], np.float32).reshape(1, 2)},
            },
        }
        o = preprocess_observation(gym)
        o["task"] = [self.instruction]
        o = self.env_pre(o)
        o = self.preprocessor(o)
        with self.torch.inference_mode():
            a = self.postprocessor(self.policy.select_action(o)).to("cpu").float().numpy()
        return (a[0] if a.ndim == 2 else a).astype(np.float32)
