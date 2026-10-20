"""VLA のスキルカタログ（PARC2026 の解析資産から生成した data/skill_catalog.json）。

1 スキル = VLA が学習時に見た正準指示文 1 つ。VLA はこの文字列で呼ぶと最もよく動く
（PARC2026 では指示文を小文字化するだけで本番スコアが 0.179 → 0.382 に上がった）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "skill_catalog.json"


@dataclass(frozen=True)
class Skill:
    id: str
    instruction: str
    suite: str
    scene_id: str
    bddl_file: str
    goal: tuple[tuple[str, ...], ...]
    obj_of_interest: tuple[str, ...]
    paraphrases: tuple[str, ...] = field(repr=False)


class SkillCatalog:
    def __init__(self, skills: list[Skill]) -> None:
        self.skills = skills
        self.by_id = {s.id: s for s in skills}
        self.by_instruction = {s.instruction: s for s in skills}

    @classmethod
    def load(cls, path: str | Path = CATALOG_PATH) -> "SkillCatalog":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        skills = [
            Skill(
                id=s["id"],
                instruction=s["instruction"],
                suite=s["suite"],
                scene_id=s["scene_id"],
                bddl_file=s["bddl_file"],
                goal=tuple(tuple(g) for g in s["goal"]),
                obj_of_interest=tuple(s["obj_of_interest"]),
                paraphrases=tuple(s["paraphrases"]),
            )
            for s in raw["skills"]
        ]
        return cls(skills)

    def in_scene(self, scene_id: str) -> list[Skill]:
        return [s for s in self.skills if s.scene_id == scene_id]

    def resolve(self, text: str) -> Skill | None:
        """LLM が返した文字列をスキルへ引き当てる（ID・正準文のどちらでも可）。"""
        key = text.strip().lower().rstrip(".")
        return self.by_id.get(text.strip()) or self.by_instruction.get(key)
