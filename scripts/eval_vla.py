"""VLA に渡す「言葉」を変えたときの成功率を測る（シミュレータ側の環境で実行）。

条件（scripts/eval_planner.py が vla_instructions.json に書き出す）:
  canonical               : 学習時の正準文（上限の目安）
  libero_plus_paraphrase  : LIBERO-plus の英語の言い換えをそのまま渡す
  plus_kotoba_bridge      : 同じ言い換えを言語ブリッジで正準文に直してから渡す
  ja_direct_translate     : 日本語の依頼を LLM で英訳しただけの文を渡す
  ja_kotoba_bridge        : 日本語の依頼を言語ブリッジで正準文に直してから渡す

推論は乱数を固定しているので、(指示文, スキル, 初期状態) が同じなら結果も同じになる
（PARC2026 で確認した決定論性）。同じ組は 1 回だけ実行して結果を共有する。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kotoba_vla.catalog import SkillCatalog  # noqa: E402
from kotoba_vla.sim.libero_scene import LiberoScene, setup_libero_config  # noqa: E402
from kotoba_vla.sim.runner import run_skill  # noqa: E402
from kotoba_vla.vla.client import VLAClient  # noqa: E402


def norm(s: str) -> str:
    """VLA サーバと同じ正規化（小文字化・空白・動詞形）。キャッシュのキーに使う。"""
    t = " ".join(str(s or "").split()).lower()
    return t.replace("pick the ", "pick up the ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instructions", type=Path, default=ROOT / "outputs" / "eval" / "vla_instructions.json")
    ap.add_argument("--results", type=Path, default=ROOT / "outputs" / "eval" / "vla_results.jsonl")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--libero-root", default=os.environ.get("LIBERO_ROOT", str(ROOT / "third_party" / "hf_libero")))
    ap.add_argument("--inits", default="0,1,2", help="使う初期状態の番号（カンマ区切り）")
    ap.add_argument("--canonical-inits", default="0,1,2", help="canonical 条件の初期状態（既定は他条件と同じにして対応比較する）")
    ap.add_argument("--conditions", default="")
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()

    setup_libero_config(args.libero_root)
    catalog = SkillCatalog.load()
    rows = json.loads(args.instructions.read_text())
    if args.conditions:
        keep = set(args.conditions.split(","))
        rows = [r for r in rows if r["condition"] in keep]

    done: dict[tuple, dict] = {}
    if args.results.exists():
        for line in args.results.read_text().splitlines():
            r = json.loads(line)
            done[(r["key_instruction"], r["goal_skill"], r["init"])] = r

    # 実行する組を作る（同じ組は 1 回だけ）
    jobs: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for r in rows:
        inits = args.canonical_inits if r["condition"] == "canonical" else args.inits
        for i in map(int, inits.split(",")):
            key = (norm(r["instruction"]), r["goal_skill"], i)
            if key not in done and (key[0], i) not in jobs[r["goal_skill"]]:
                jobs[r["goal_skill"]].append((key[0], i))

    client = VLAClient(args.server)
    print("server:", client.health())
    n_total = sum(len(v) for v in jobs.values())
    print(f"{n_total} episodes to run ({len(done)} cached)")
    n = 0
    with args.results.open("a") as out:
        for sid, todo in jobs.items():
            skill = catalog.by_id[sid]
            scene = LiberoScene(skill, video_size=256)
            others = [s for s in catalog.in_scene(skill.scene_id) if s.id != sid]
            for text, init in todo:
                t0 = time.perf_counter()
                scene.reset(init)
                before = {s.id for s in others if scene.goal_reached(s)}
                if not text:  # 計画が空（ブリッジがスキルを選べなかった）なら失敗として記録
                    o = None
                else:
                    o = run_skill(scene, client, skill, vla_instruction=text, max_steps=args.max_steps,
                                  max_attempts=1)
                # 頼んでいない別スキルのゴールを新たに満たしたか（＝言葉の取り違えで別の作業をした）
                other_done = sorted(s.id for s in others if s.id not in before and scene.goal_reached(s))
                rec = {
                    "key_instruction": text, "goal_skill": sid, "init": init,
                    "success": bool(o and o.success), "clean_success": bool(o and o.clean_success),
                    "steps": o.steps if o else 0, "disturbance_mm": o.disturbance_mm if o else 0.0,
                    "disturbed_object": o.disturbed_object if o else "", "other_goals": other_done,
                    "wall_s": round(time.perf_counter() - t0, 1),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                done[(text, sid, init)] = rec
                n += 1
                print(f"[{n}/{n_total}] {'OK' if rec['success'] else 'NG'} {sid} init={init} "
                      f"steps={rec['steps']} {rec['wall_s']}s  {text!r}", flush=True)
            scene.close()

    # 条件ごとの集計
    table = defaultdict(lambda: {"n": 0, "success": 0, "clean": 0, "fail_did_other_task": 0})
    for r in rows:
        inits = args.canonical_inits if r["condition"] == "canonical" else args.inits
        for i in map(int, inits.split(",")):
            rec = done.get((norm(r["instruction"]), r["goal_skill"], i))
            if rec is None:
                continue
            t = table[r["condition"]]
            t["n"] += 1
            t["success"] += rec["success"]
            t["clean"] += rec["clean_success"]
            t["fail_did_other_task"] += (not rec["success"]) and bool(rec.get("other_goals"))
    summary = {c: {**v, "success_rate": round(v["success"] / v["n"], 3), "clean_rate": round(v["clean"] / v["n"], 3)}
               for c, v in table.items() if v["n"]}
    (args.results.parent / "vla_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
