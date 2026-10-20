"""計画（スキルの列）を VLA で順に実行し、各スキルを検証する実行エンジン。

1 スキルの流れ:
  1. VLA に正準指示文を渡して /reset
  2. 観測を送って行動を受け取り、環境を 1 ステップ進める（20Hz）。毎ステップ BDDL ゴールを判定
  3. ゴール達成 → 成功。上限ステップに達したら失敗
  4. 対象外物体の変位（PARC2026 の 1mm 判定）を記録
  5. 次のスキルへ移る前に、アームを初期姿勢へ戻す（VLA は初期姿勢から学習しているため）
  失敗したらホームへ戻ってから同じスキルを再試行する（最大 max_attempts 回）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from ..catalog import Skill
from .libero_scene import DISTURB_THRESHOLD_M, LiberoScene


@dataclass
class SkillOutcome:
    skill_id: str
    vla_instruction: str
    success: bool
    attempts: int
    steps: int
    disturbance_mm: float
    disturbed_object: str
    already_done: bool = False
    vla_time_s: float = 0.0
    attempt_log: list[dict] = field(default_factory=list)

    @property
    def clean_success(self) -> bool:
        """PARC2026 の採点と同じ「成功し、かつ対象外の物体を 1mm 以上動かしていない」。"""
        return self.success and self.disturbance_mm <= DISTURB_THRESHOLD_M * 1000

    def to_dict(self) -> dict:
        return {**asdict(self), "clean_success": self.clean_success}


# フレームごとに呼ばれるフック: (scene, info) -> None
FrameHook = Callable[[LiberoScene, dict], None]


def _axis_angle_error(q_target: np.ndarray, q_now: np.ndarray) -> np.ndarray:
    """(x, y, z, w) の 2 姿勢の差をワールド座標の回転ベクトルで返す。"""
    from scipy.spatial.transform import Rotation as R

    return (R.from_quat(q_target) * R.from_quat(q_now).inv()).as_rotvec()


def return_home(scene: LiberoScene, hook: FrameHook | None = None, info: dict | None = None,
                max_steps: int = 90) -> int:
    """グリッパを開き、少し持ち上げてから初期姿勢へ戻す（OSC の差分指令による P 制御）。"""
    info = dict(info or {}, phase="home")
    steps = 0

    def drive(target_pos, target_quat, n, tol=0.01):
        nonlocal steps
        for _ in range(n):
            pos = scene.obs["robot0_eef_pos"]
            dp = target_pos - pos
            dr = _axis_angle_error(target_quat, scene.obs["robot0_eef_quat"])
            if np.linalg.norm(dp) < tol and np.linalg.norm(dr) < 0.05:
                return
            a = np.zeros(7)
            a[:3] = np.clip(dp / 0.05 * 0.5, -1, 1)
            a[3:6] = np.clip(dr / 0.5 * 0.5, -1, 1)
            a[6] = -1.0
            scene.step(a)
            steps += 1
            if hook:
                hook(scene, info)

    pos0 = scene.obs["robot0_eef_pos"].copy()
    quat0 = scene.obs["robot0_eef_quat"].copy()
    drive(pos0, quat0, 6)  # まず開く
    lift = pos0.copy()
    lift[2] = max(pos0[2] + 0.06, scene.home_eef_pos[2] - 0.02)
    drive(lift, quat0, 25)
    drive(scene.home_eef_pos, scene.home_eef_quat, max_steps)
    return steps


def run_skill(scene: LiberoScene, client, skill: Skill, vla_instruction: str | None = None,
              max_steps: int = 300, max_attempts: int = 2, hook: FrameHook | None = None,
              info: dict | None = None, home_between_attempts: bool = True) -> SkillOutcome:
    """1 スキルを実行する。vla_instruction を変えると「VLA に渡す文」だけを差し替えられる（比較実験用）。"""
    text = vla_instruction if vla_instruction is not None else skill.instruction
    info = dict(info or {})
    ignore = set(skill.obj_of_interest)
    outcome = SkillOutcome(skill.id, text, False, 0, 0, 0.0, "")

    if scene.goal_reached(skill):  # 既に満たされているなら何もしない
        outcome.success, outcome.already_done = True, True
        return outcome

    vla_time = 0.0
    for attempt in range(1, max_attempts + 1):
        outcome.attempts = attempt
        if attempt > 1 and home_between_attempts:
            return_home(scene, hook, {**info, "attempt": attempt})
        start = scene.object_positions()
        sent = client.reset(text)
        reached = False
        worst, worst_name = 0.0, ""
        t_attempt = 0
        for t in range(max_steps):
            t0 = time.perf_counter()
            action = client.act(scene.obs)
            vla_time += time.perf_counter() - t0
            scene.step(action)
            t_attempt += 1
            d, name = scene.max_disturbance(start, scene.object_positions(), ignore)
            if d > worst:
                worst, worst_name = d, name
            reached = scene.goal_reached(skill)
            if hook:
                hook(scene, {**info, "phase": "vla", "attempt": attempt, "t": t, "vla_text": sent,
                             "action": np.asarray(action).tolist(), "goal": reached, "disturb_mm": worst * 1000})
            if reached:
                break
        outcome.steps += t_attempt
        outcome.attempt_log.append({"attempt": attempt, "steps": t_attempt, "success": reached,
                                    "disturbance_mm": round(worst * 1000, 2), "disturbed_object": worst_name})
        if worst * 1000 > outcome.disturbance_mm:
            outcome.disturbance_mm, outcome.disturbed_object = round(worst * 1000, 2), worst_name
        if reached:
            outcome.success = True
            break
    outcome.vla_time_s = round(vla_time, 2)
    return outcome


def run_plan(scene: LiberoScene, client, skills: list[Skill], hook: FrameHook | None = None,
             max_steps: int = 300, max_attempts: int = 2, stop_on_failure: bool = True) -> list[SkillOutcome]:
    """スキルの列を順に実行する。前のスキルが失敗したら（既定では）そこで止める。"""
    outcomes = []
    for i, skill in enumerate(skills):
        info = {"skill_index": i, "skill_id": skill.id}
        if i > 0:
            return_home(scene, hook, info)
        out = run_skill(scene, client, skill, max_steps=max_steps, max_attempts=max_attempts, hook=hook, info=info)
        outcomes.append(out)
        if hook:
            hook(scene, {**info, "phase": "verified", "success": out.success, "disturb_mm": out.disturbance_mm})
        if not out.success and stop_on_failure:
            break
    return outcomes
