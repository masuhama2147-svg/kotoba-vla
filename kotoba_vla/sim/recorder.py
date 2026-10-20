"""実行中のフレームと状態を記録する（動画合成と事後解析のための生データ）。

- main.mp4  : 動画用カメラの高解像度映像（1 ステップ = 1 フレーム、20Hz）
- agent.mp4 / wrist.mp4 : 正面・手首カメラ（VLA への入力画像そのもの、256x256）
- timeline.jsonl : 各フレームのスキル番号・フェーズ・VLA に渡した文・ゴール判定・変位など
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .libero_scene import LiberoScene


class FfmpegWriter:
    def __init__(self, path: Path, size: tuple[int, int], fps: int = 20, crf: int = 14) -> None:
        w, h = size
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


class RunRecorder:
    def __init__(self, out_dir: str | Path, video_size: int = 720, fps: int = 20) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.video_size = video_size
        self.main = FfmpegWriter(self.dir / "main.mp4", (video_size, video_size), fps)
        self.agent = FfmpegWriter(self.dir / "agent.mp4", (256, 256), fps)
        self.wrist = FfmpegWriter(self.dir / "wrist.mp4", (256, 256), fps)
        self.timeline = (self.dir / "timeline.jsonl").open("w")
        self.n = 0

    def __call__(self, scene: LiberoScene, info: dict) -> None:
        if info.get("phase") == "verified":  # 検証結果は映像フレームを伴わないイベントとして残す
            self.timeline.write(json.dumps({"frame": self.n - 1, "event": "verified", **info}) + "\n")
            return
        self.main.write(scene.render(size=self.video_size))
        self.agent.write(scene.agent_image())
        self.wrist.write(scene.wrist_image())
        rec = {"frame": self.n, **{k: v for k, v in info.items() if k != "action"}}
        if "action" in info:
            rec["gripper"] = round(float(info["action"][6]), 3)
        self.timeline.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n += 1

    def snapshot(self, scene: LiberoScene, info: dict, repeat: int = 1) -> None:
        """静止したシーンをフレームとして足す（計画中などの「待ち」区間に使う）。"""
        for _ in range(repeat):
            self(scene, {**info, "phase": info.get("phase", "idle")})

    def close(self) -> None:
        self.main.close()
        self.agent.close()
        self.wrist.close()
        self.timeline.close()
