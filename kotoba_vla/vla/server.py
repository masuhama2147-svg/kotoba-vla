"""VLA 推論サーバ（ROS 2 のサービスに相当する、モデル専用プロセス）。

π0.5 は lerobot 0.5.2 / transformers 5.3 / torch 2.8 を要し、シミュレータ側
（robosuite 1.4 / numpy 1.26 / Python 3.10）と同じ Python 環境に同居できない（PARC2026 で実測）。そこでモデルだけを別プロセスに切り出し、
HTTP で観測を受けて行動を返す。

  GET  /health                     -> {"status": "ok", "model": ...}
  POST /reset   {"instruction": s} -> エピソード（スキル）開始。内部状態と乱数を初期化
  POST /act     msgpack 観測       -> msgpack {"action": float32[7]}

起動:
  .venv-vla/bin/python -m kotoba_vla.vla.server --model-dir weights/pi05_parc_t1 --hf-home weights/hf_cache
  .venv-vla/bin/python -m kotoba_vla.vla.server --dummy     # 重み無しで配線だけ確かめる
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, Response

from .protocol import decode_arrays, encode_arrays

_LOG = logging.getLogger("kotoba.server")


class DummyPolicy:
    """重み無しで配線を確かめるための方策（その場で静止し、グリッパは開いたまま）。"""

    def reset(self, instruction: str = "", seed=None) -> None:
        self.instruction = instruction

    def get_action(self, obs: dict) -> np.ndarray:
        return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


def build_app(policy, model_name: str):
    app = FastAPI(title="kotoba-vla policy server")
    stats = {"act_calls": 0, "max_latency": 0.0, "n_inference": 0, "inference_s_total": 0.0}

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, **stats}

    @app.post("/reset")
    async def reset(request: Request):
        body = await request.body()
        instruction = json.loads(body).get("instruction", "") if body else ""
        seed = json.loads(body).get("seed") if body else None
        policy.reset(instruction=instruction, seed=seed)
        return {"status": "ok", "instruction": getattr(policy, "instruction", instruction)}

    @app.post("/act")
    async def act(request: Request):
        t0 = time.perf_counter()
        obs = decode_arrays(await request.body())
        action = np.asarray(policy.get_action(obs), dtype=np.float32)
        dt = time.perf_counter() - t0
        stats["act_calls"] += 1
        stats["max_latency"] = max(stats["max_latency"], dt)
        if dt > 0.1:  # 行動チャンクを推論したリクエスト（残りはキューから取り出すだけで 1ms 程度）
            stats["n_inference"] += 1
            stats["inference_s_total"] += dt
        return Response(content=encode_arrays({"action": action}), media_type="application/x-msgpack")

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", help="π0.5 の重みディレクトリ（model.safetensors と config.json を含む）")
    ap.add_argument("--hf-home", help="PaliGemma トークナイザを含む HF キャッシュ")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dummy:
        policy, name = DummyPolicy(), "dummy"
    else:
        from .pi05_t1 import Pi05T1Policy

        policy = Pi05T1Policy(args.model_dir, device=args.device, hf_home=args.hf_home)
        name = f"pi05:{Path(args.model_dir).name}"
        _LOG.info("loaded %s in %.1fs (verified %d tensors, n_action_steps=%d)",
                  name, policy.load_s, policy.n_verified, policy.n_action_steps)
        t0 = time.perf_counter()  # 初回推論の遅延初期化をここで済ませる
        policy.reset("warm up", seed=0)
        policy.get_action({
            "agentview_image": np.zeros((256, 256, 3), np.uint8),
            "robot0_eye_in_hand_image": np.zeros((256, 256, 3), np.uint8),
            "robot0_joint_pos": np.zeros(7), "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0.0, 0.7071068, 0.0, 0.7071068]), "robot0_gripper_qpos": np.zeros(2),
        })
        policy.reset("")
        _LOG.info("warm-up %.1fs", time.perf_counter() - t0)

    import uvicorn

    uvicorn.run(build_app(policy, name), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
