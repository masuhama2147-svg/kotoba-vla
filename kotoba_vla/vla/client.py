"""VLA 推論サーバのクライアント（シミュレータ側の Python 環境から呼ぶ。標準ライブラリ + msgpack のみ）。"""

from __future__ import annotations

import json
import urllib.request

import numpy as np

from .protocol import decode_arrays, encode_arrays

# サーバへ送る観測キー（PARC2026 の評価系と同じ 6 キー。物体の真の位置は送らない）
OBS_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_joint_pos",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


class VLAClient:
    def __init__(self, url: str = "http://127.0.0.1:8000", timeout: float = 60.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, data: bytes, content_type: str) -> bytes:
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": content_type})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()

    def health(self) -> dict:
        with urllib.request.urlopen(self.url + "/health", timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def reset(self, instruction: str, seed: int = 0) -> str:
        body = json.dumps({"instruction": instruction, "seed": seed}).encode()
        out = json.loads(self._post("/reset", body, "application/json"))
        return out.get("instruction", instruction)

    def act(self, obs: dict) -> np.ndarray:
        payload = encode_arrays({k: np.asarray(obs[k]) for k in OBS_KEYS})
        return decode_arrays(self._post("/act", payload, "application/x-msgpack"))["action"]
