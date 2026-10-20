"""VLA 評価（scripts/eval_vla.py）で使う指示文の一覧を作る（AI 側の環境で実行）。

条件ごとに、同じスキル・同じ初期状態で「VLA に渡す文」だけを変える:
  canonical               : 学習時の正準文
  libero_plus_paraphrase  : LIBERO-plus の英語の言い換え（スキルごとに 3 文）をそのまま
  plus_kotoba_bridge      : 同じ 3 文を言語ブリッジに通した結果
                            （公平のため、この 3 文は検索キーと LLM への例文から外す）
  ja_direct_translate     : 日本語の依頼（開発セット 3 文）を LLM で英訳しただけ
  ja_kotoba_bridge        : 同じ日本語の依頼を言語ブリッジに通した結果
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kotoba_vla.bridge.planner import LocalLLM, SkillPlanner  # noqa: E402
from kotoba_vla.bridge.retriever import SkillRetriever  # noqa: E402
from kotoba_vla.catalog import SkillCatalog  # noqa: E402

SCENE = "libero_goal"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=ROOT / "outputs" / "eval" / "planner_rows_dev.json")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval" / "vla_instructions.json")
    ap.add_argument("--n-plus", type=int, default=3)
    args = ap.parse_args()

    catalog = SkillCatalog.load()
    goal = catalog.in_scene(SCENE)
    rng = random.Random(0)
    plus = {s.id: rng.sample(list(s.paraphrases), args.n_plus) for s in goal}
    held_out = {p for ps in plus.values() for p in ps}
    planner = SkillPlanner(catalog, SkillRetriever(catalog, exclude=held_out), LocalLLM())
    rows = [r for r in json.loads(args.rows.read_text()) if r["kind"] == "single" and r["source"] == "text"]

    out = []
    for s in goal:
        out.append({"goal_skill": s.id, "condition": "canonical", "variant": 0, "instruction": s.instruction})
        for k, p in enumerate(plus[s.id]):
            out.append({"goal_skill": s.id, "condition": "libero_plus_paraphrase", "variant": k, "instruction": p})
            pl = planner.plan(p, SCENE)
            out.append({"goal_skill": s.id, "condition": "plus_kotoba_bridge", "variant": k, "source_text": p,
                        "planned": [st.skill.id for st in pl.steps],
                        "instruction": pl.steps[0].skill.instruction if pl.steps else ""})
            print(f"[plus] {p} -> {out[-1]['planned']}", flush=True)
        for r in rows:
            if r["skill"] != s.id:
                continue
            out.append({"goal_skill": s.id, "condition": "ja_direct_translate", "variant": r["variant"],
                        "ja": r["text"], "instruction": r["direct_translate"]})
            planned = r["llm_planner"]
            out.append({"goal_skill": s.id, "condition": "ja_kotoba_bridge", "variant": r["variant"], "ja": r["text"],
                        "planned": planned, "instruction": catalog.by_id[planned[0]].instruction if planned else ""})
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    by_cond = {}
    for o in out:
        by_cond[o["condition"]] = by_cond.get(o["condition"], 0) + 1
    print(by_cond, "->", args.out)


if __name__ == "__main__":
    main()
