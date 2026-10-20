"""複数ステップの日本語依頼を、1 つのシーンで最後までやり切れるかを測る（シミュレータ側の環境）。

計画は eval_planner.py の結果（文字入力のときの LLM 計画）を使う。比較:
  full  : スキルの間で初期姿勢へ戻す ＋ 失敗したら 1 回だけ再試行（本システム）
  naive : 戻さず・再試行なしで、前のスキルの終わった姿勢から次のスキルを始める
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kotoba_vla.catalog import SkillCatalog  # noqa: E402
from kotoba_vla.sim.libero_scene import LiberoScene, setup_libero_config  # noqa: E402
from kotoba_vla.sim.runner import return_home, run_skill  # noqa: E402
from kotoba_vla.vla.client import VLAClient  # noqa: E402


def run_chain(scene, client, skills, mode: str, max_steps: int) -> list[dict]:
    outs = []
    for i, sk in enumerate(skills):
        if i > 0 and mode == "full":
            return_home(scene)
        o = run_skill(scene, client, sk, max_steps=max_steps, max_attempts=2 if mode == "full" else 1,
                      home_between_attempts=True)
        outs.append(o.to_dict())
        if not o.success:
            break
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=ROOT / "outputs" / "eval" / "planner_rows_dev.json")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval" / "chain_results.json")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--libero-root", default=os.environ.get("LIBERO_ROOT", str(ROOT / "third_party" / "hf_libero")))
    ap.add_argument("--inits", default="0,1,2")
    ap.add_argument("--modes", default="full,naive")
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()

    setup_libero_config(args.libero_root)
    catalog = SkillCatalog.load()
    rows = [r for r in json.loads(args.rows.read_text()) if r["kind"] == "multi" and r["source"] == "text"]
    client = VLAClient(args.server)
    results = json.loads(args.out.read_text()) if args.out.exists() else []
    seen = {(r["text"], r["mode"], r["init"]) for r in results}

    for r in rows:
        skills = [catalog.by_id[s] for s in r["llm_planner"]]
        if not skills:
            continue
        scene = LiberoScene(skills[0], video_size=256)
        for mode in args.modes.split(","):
            for init in map(int, args.inits.split(",")):
                if (r["text"], mode, init) in seen:
                    continue
                t0 = time.perf_counter()
                scene.reset(init)
                outs = run_chain(scene, client, skills, mode, args.max_steps)
                ok = len(outs) == len(skills) and all(o["success"] for o in outs)
                rec = {"text": r["text"], "plan": r["llm_planner"], "plan_correct": r["llm_planner"] == r["expect"],
                       "mode": mode, "init": init, "chain_success": ok,
                       "n_done": sum(o["success"] for o in outs), "n_skills": len(skills),
                       "outcomes": outs, "wall_s": round(time.perf_counter() - t0, 1)}
                results.append(rec)
                args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
                print(f"{'OK' if ok else 'NG'} [{mode}] init={init} {rec['n_done']}/{len(skills)} {r['text']}", flush=True)
        scene.close()

    summary = {}
    for mode in args.modes.split(","):
        rs = [r for r in results if r["mode"] == mode]
        if rs:
            summary[mode] = {"n": len(rs), "chain_success": sum(r["chain_success"] for r in rs),
                             "skill_success": sum(r["n_done"] for r in rs), "skills_total": sum(r["n_skills"] for r in rs)}
    (args.out.parent / "chain_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
