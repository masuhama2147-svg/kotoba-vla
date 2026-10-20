"""デモ 2 の素材：同じ依頼・同じ初期状態で「英訳しただけの文」と「正準文」を VLA に渡し、両方を収録する。

評価結果（results/vla_results.jsonl）から、英訳では失敗して正準文では成功した組を選ぶ。
別の作業をしてしまった失敗（PARC で見た「別の器を運ぶ」型）を優先する。

  .venv-sim/bin/python scripts/record_contrast.py --server http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_vla import norm  # noqa: E402

from kotoba_vla.catalog import SkillCatalog  # noqa: E402
from kotoba_vla.sim.libero_scene import LiberoScene, setup_libero_config  # noqa: E402
from kotoba_vla.sim.recorder import RunRecorder  # noqa: E402
from kotoba_vla.sim.runner import run_skill  # noqa: E402
from kotoba_vla.vla.client import VLAClient  # noqa: E402


def pick(instructions: list[dict], results: dict) -> dict:
    cands = []
    for r in instructions:
        if r["condition"] != "ja_direct_translate":
            continue
        for init in (0, 1, 2):
            raw = results.get((norm(r["instruction"]), r["goal_skill"], init))
            canon = next((x for x in instructions if x["condition"] == "ja_kotoba_bridge"
                          and x["goal_skill"] == r["goal_skill"] and x["variant"] == r["variant"]), None)
            if raw is None or canon is None:
                continue
            ok = results.get((norm(canon["instruction"]), r["goal_skill"], init))
            if ok and ok["success"] and not raw["success"]:
                score = (2 if raw.get("other_goals") else 0) + (1 if init == 0 else 0) - ok["steps"] / 1000
                cands.append((score, {"skill": r["goal_skill"], "init": init, "ja": r["ja"],
                                      "direct": r["instruction"], "bridge": canon["instruction"],
                                      "other_goals": raw.get("other_goals", [])}))
    if not cands:
        raise SystemExit("英訳だけ失敗・正準文で成功の組が見つからない")
    cands.sort(key=lambda x: -x[0])
    return cands[0][1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--libero-root", default=os.environ.get("LIBERO_ROOT", str(ROOT / "third_party" / "hf_libero")))
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "vla_results.jsonl")
    ap.add_argument("--instructions", type=Path, default=ROOT / "results" / "vla_instructions.json")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "runs")
    args = ap.parse_args()

    setup_libero_config(args.libero_root)
    catalog = SkillCatalog.load()
    results = {}
    for line in args.results.read_text().splitlines():
        r = json.loads(line)
        results[(r["key_instruction"], r["goal_skill"], r["init"])] = r
    choice = pick(json.loads(args.instructions.read_text()), results)
    print("選んだ組:", json.dumps(choice, ensure_ascii=False))

    skill = catalog.by_id[choice["skill"]]
    client = VLAClient(args.server)
    scene = LiberoScene(skill, video_size=720)
    for name, text in (("contrast_direct", choice["direct"]), ("contrast_bridge", choice["bridge"])):
        run = args.out / name
        if run.exists():
            shutil.rmtree(run)
        run.mkdir(parents=True)
        scene.reset(choice["init"])
        rec = RunRecorder(run, video_size=720)
        rec.snapshot(scene, {"phase": "idle"})
        o = run_skill(scene, client, skill, vla_instruction=text, max_attempts=1, hook=rec,
                      info={"skill_index": 0, "skill_id": skill.id})
        rec.snapshot(scene, {"phase": "done"})
        rec.close()
        plan = {"text": choice["ja"], "transcript": choice["ja"], "scene": skill.scene_id,
                "plan": {"steps": [{"skill_id": skill.id, "instruction": text, "why": ""}], "unsupported": "", "reply": ""}}
        (run / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
        (run / "result.json").write_text(json.dumps({"all_success": o.success, "outcomes": [o.to_dict()],
                                                     "choice": choice}, ensure_ascii=False, indent=1))
        print(f"  {name}: {'OK' if o.success else 'NG'} steps={o.steps} {text!r}")
    scene.close()


if __name__ == "__main__":
    main()
