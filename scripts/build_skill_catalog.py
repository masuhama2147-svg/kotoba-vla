"""PARC2026 で作った解析資産から、VLA のスキルカタログを作る。

入力（PARC2026 の作業ディレクトリ）:
  artifacts/paraphrase_pool.json   … 40 基底タスク × 約 51 文の言い換え（LIBERO-plus 由来、計 2,052 文）
  artifacts/libero_bddl_ref/       … LIBERO の BDDL ファイル（ゴール述語・注目物体を読む）

出力:
  data/skill_catalog.json          … スキル ID / 正準指示文 / スイート / ゴール述語 / 言い換え

使い方:
  python scripts/build_skill_catalog.py --parc-dir ../PARCコンペ/parc_final
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("libero_goal", "libero_spatial", "libero_object", "libero_10")


def find_bddl(bddl_root: Path, stem: str) -> tuple[str, Path] | None:
    for suite in SUITES:
        for p in sorted((bddl_root / suite).glob(f"*{stem}.bddl")):
            # libero_10 はファイル名の先頭にシーン名が付く（例: KITCHEN_SCENE3_turn_on_...）
            if p.stem == stem or re.fullmatch(r"[A-Z0-9_]+_" + re.escape(stem), p.stem):
                return suite, p
    return None


def parse_block(text: str, name: str) -> str:
    m = re.search(rf"\(:{name}(.*?)\n\s*\)\s*\n", text, flags=re.S)
    return m.group(1) if m else ""


def parse_bddl(path: Path) -> dict:
    text = path.read_text()
    language = re.search(r"\(:language\s+(.*?)\)", text).group(1).strip()
    goal = parse_block(text, "goal")
    atoms = re.findall(r"\((On|In|Open|Close|Turnon|Turnoff)\s+([^)]*)\)", goal)
    interest = parse_block(text, "obj_of_interest").split()
    objects = re.findall(r"^\s*(\w+)\s+-\s+\w+", parse_block(text, "objects"), flags=re.M)
    fixtures = re.findall(r"^\s*(\w+)\s+-\s+\w+", parse_block(text, "fixtures"), flags=re.M)
    return {
        "language": language,
        "goal": [[a, *args.split()] for a, args in atoms],
        "obj_of_interest": interest,
        "objects": objects,
        "fixtures": fixtures,
        "scene_signature": " ".join(sorted(objects + fixtures)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parc-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "skill_catalog.json")
    args = ap.parse_args()

    pool = json.loads((args.parc_dir / "artifacts" / "paraphrase_pool.json").read_text())
    bddl_root = args.parc_dir / "artifacts" / "libero_bddl_ref"

    skills = []
    for instruction in pool["train_tasks"]:
        stem = pool["task_to_base_stem"][instruction]
        found = find_bddl(bddl_root, stem)
        if found is None:
            raise SystemExit(f"BDDL が見つからない: {stem}")
        suite, path = found
        info = parse_bddl(path)
        skills.append(
            {
                "id": stem,
                "instruction": instruction,
                "suite": suite,
                "bddl_file": path.name,
                "goal": info["goal"],
                "obj_of_interest": info["obj_of_interest"],
                "scene_signature": info["scene_signature"],
                "paraphrases": sorted(set(pool["paraphrase_pool"][instruction])),
            }
        )

    # LIBERO-Goal の 10 スキルは物体・配置が同一でゴールだけが違う → 1 シーンで連続実行できる。
    # libero_spatial も物体の種類は同じだが配置がタスクごとに違うので、同じシーンとは扱わない。
    goal_sigs = {s["scene_signature"] for s in skills if s["suite"] == "libero_goal"}
    assert len(goal_sigs) == 1, "LIBERO-Goal の物体構成が一致していない"
    scenes: dict[str, list[str]] = {}
    for s in skills:
        s["scene_id"] = "libero_goal" if s["suite"] == "libero_goal" else s["id"]
        scenes.setdefault(s["scene_id"], []).append(s["id"])
        del s["scene_signature"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"source": "PARC2026 artifacts (LIBERO / LIBERO-plus)", "skills": skills},
                                   ensure_ascii=False, indent=1))
    n_para = sum(len(s["paraphrases"]) for s in skills)
    print(f"{len(skills)} skills, {n_para} paraphrases, {len(scenes)} scenes -> {args.out}")
    for sid, members in scenes.items():
        if len(members) > 1:
            print(f"  multi-skill scene ({len(members)}): {members[:3]} ...")


if __name__ == "__main__":
    main()
