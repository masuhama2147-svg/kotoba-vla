"""LIBERO（robosuite / MuJoCo）のシーンを、スキルを連続実行できる形で包む。

- 1 シーンに複数のスキル（ゴール）を順に与えられる。LIBERO-Goal の 10 スキルは物体・配置が
  共通なので、ゴール述語だけを差し替えて同じ環境で判定する。
- 成功判定は BDDL のゴール述語（LIBERO 本来の判定）をそのまま使う。
- 安全指標として PARC2026 の採点と同じ「対象外の物体が 1mm を超えて動いたか」を測る。

このモジュールはシミュレータ側の環境（conda env `parc310`: robosuite 1.4 / mujoco 3.7）で動く。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..catalog import Skill

ROOT = Path(__file__).resolve().parents[2]
DISTURB_THRESHOLD_M = 0.001  # PARC2026 の衝突判定と同じ 1mm（3 軸の絶対変位の和）


def setup_libero_config(libero_root: str | Path) -> None:
    """~/.libero を汚さず、このプロジェクト専用の LIBERO 設定を使う。"""
    libero_root = Path(libero_root).resolve()
    pkg = libero_root / "libero" / "libero"
    cfg_dir = ROOT / ".libero"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "config.yaml").write_text(
        "\n".join(
            f"{k}: {v}"
            for k, v in {
                "benchmark_root": pkg,
                "bddl_files": pkg / "bddl_files",
                "init_states": pkg / "init_files",
                "datasets": libero_root / "libero" / "datasets",
                "assets": pkg / "assets",
            }.items()
        )
        + "\n"
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg_dir)


@dataclass
class StepResult:
    obs: dict
    goal_reached: bool


class LiberoScene:
    def __init__(self, base_skill: Skill, camera_size: int = 256, video_size: int = 720,
                 video_camera: str = "frontview") -> None:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.base_skill = base_skill
        bddl = Path(get_libero_path("bddl_files")) / base_skill.suite / base_skill.bddl_file
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=camera_size, camera_widths=camera_size
        )
        suite = benchmark.get_benchmark_dict()[base_skill.suite]()
        names = [suite.get_task(i).name for i in range(suite.n_tasks)]
        task_id = names.index(Path(base_skill.bddl_file).stem)
        self.init_states = suite.get_task_init_states(task_id)
        self.video_size = video_size
        self.video_camera = video_camera
        self.obs: dict = {}
        self.home_eef_pos: np.ndarray | None = None
        self.home_eef_quat: np.ndarray | None = None

    @property
    def inner(self):
        return self.env.env

    # ---------------------------------------------------------------- 基本操作
    def reset(self, init_index: int = 0, settle_steps: int = 10) -> dict:
        self.env.seed(0)
        self.env.reset()
        self.obs = self.env.set_init_state(self.init_states[init_index % len(self.init_states)])
        # 物体を落ち着かせる（LIBERO の標準的な評価手順と同じく、静止行動を数ステップ）
        for _ in range(settle_steps):
            self.obs, _, _, _ = self.env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
        self.home_eef_pos = self.obs["robot0_eef_pos"].copy()
        self.home_eef_quat = self.obs["robot0_eef_quat"].copy()
        return self.obs

    def step(self, action: np.ndarray) -> dict:
        self.obs, _, _, _ = self.env.step(np.asarray(action, dtype=np.float64))
        return self.obs

    # ---------------------------------------------------------------- ゴール判定
    def goal_reached(self, skill: Skill) -> bool:
        """スキルのゴール述語（BDDL）がすべて成り立っているか。"""
        inner = self.inner
        saved = inner.parsed_problem["goal_state"]
        try:
            inner.parsed_problem["goal_state"] = [self._as_libero_atom(a) for a in skill.goal]
            return bool(inner._check_success())
        finally:
            inner.parsed_problem["goal_state"] = saved

    def _as_libero_atom(self, atom) -> list[str]:
        # LIBERO のパーサと同じ表記（述語は小文字）に合わせる
        return [atom[0].lower(), *atom[1:]]

    # ---------------------------------------------------------------- 安全指標
    def object_positions(self) -> dict[str, np.ndarray]:
        return {
            k[: -len("_pos")]: np.asarray(v).copy()
            for k, v in self.obs.items()
            if k.endswith("_pos") and not k.startswith("robot0") and "_to_robot0" not in k
        }

    @staticmethod
    def max_disturbance(start: dict[str, np.ndarray], now: dict[str, np.ndarray], ignore: set[str]) -> tuple[float, str]:
        """対象外の物体の最大 L1 変位 [m] と、その物体名。"""
        worst, name = 0.0, ""
        for k, p0 in start.items():
            if k in ignore or k not in now:
                continue
            d = float(np.abs(now[k] - p0).sum())
            if d > worst:
                worst, name = d, k
        return worst, name

    # ---------------------------------------------------------------- 描画
    def render(self, camera: str | None = None, size: int | None = None) -> np.ndarray:
        """動画用の高解像度画像（上下を正しい向きに直して返す）。"""
        size = size or self.video_size
        img = self.inner.sim.render(camera_name=camera or self.video_camera, height=size, width=size)
        return np.ascontiguousarray(img[::-1])

    def wrist_image(self) -> np.ndarray:
        return np.ascontiguousarray(self.obs["robot0_eye_in_hand_image"][::-1])

    def agent_image(self) -> np.ndarray:
        return np.ascontiguousarray(self.obs["agentview_image"][::-1])

    def close(self) -> None:
        self.env.close()
