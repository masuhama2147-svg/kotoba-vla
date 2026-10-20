"""plan.json のスキル列を LIBERO 上で VLA に実行させる（シミュレータ側の環境 parc310 で実行）。

  conda run -n parc310 python scripts/execute_plan.py --run outputs/runs/demo1 --init 0

前提: VLA 推論サーバ（kotoba_vla/vla/server.py）が起動していること。
出力: main.mp4 / wrist.mp4 / timeline.jsonl / result.json（run ディレクトリ内）
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
from kotoba_vla.sim.recorder import RunRecorder  # noqa: E402
from kotoba_vla.sim.runner import run_plan  # noqa: E402
from kotoba_vla.vla.client import VLAClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--libero-root", default=os.environ.get("LIBERO_ROOT", str(ROOT / "third_party" / "hf_libero")))
    ap.add_argument("--init", type=int, default=0, help="LIBERO の初期状態の番号（0〜49）")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--video-size", type=int, default=720)
    ap.add_argument("--camera", default="frontview")
    args = ap.parse_args()

    setup_libero_config(args.libero_root)
    plan = json.loads((args.run / "plan.json").read_text())
    catalog = SkillCatalog.load()
    skills = [catalog.by_id[s["skill_id"]] for s in plan["plan"]["steps"]]
    if not skills:  # 計画が空（できない依頼）でも、シーンの映像だけは残す
        scene = LiberoScene(catalog.in_scene(plan.get("scene", "libero_goal"))[0], video_size=args.video_size,
                            video_camera=args.camera)
        scene.reset(args.init)
        rec = RunRecorder(args.run, video_size=args.video_size)
        rec.snapshot(scene, {"phase": "idle"}, repeat=2)
        rec.close()
        scene.close()
        print("実行するスキルがありません（計画が空）")
        (args.run / "result.json").write_text(json.dumps({"outcomes": [], "all_success": False}, indent=1))
        return

    client = VLAClient(args.server)
    print("server:", client.health())
    scene = LiberoScene(skills[0], video_size=args.video_size, video_camera=args.camera)
    scene.reset(args.init)
    rec = RunRecorder(args.run, video_size=args.video_size)
    rec.snapshot(scene, {"phase": "idle"})

    t0 = time.perf_counter()
    outcomes = run_plan(scene, client, skills, hook=rec, max_steps=args.max_steps, max_attempts=args.max_attempts)
    wall = time.perf_counter() - t0
    rec.snapshot(scene, {"phase": "done"})
    rec.close()
    scene.close()

    result = {
        "init": args.init,
        "wall_time_s": round(wall, 1),
        "frames": rec.n,
        "all_success": len(outcomes) == len(skills) and all(o.success for o in outcomes),
        "outcomes": [o.to_dict() for o in outcomes],
    }
    (args.run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for o in outcomes:
        print(f"  {'OK ' if o.success else 'NG '} {o.vla_instruction!r} steps={o.steps} attempts={o.attempts} "
              f"disturb={o.disturbance_mm}mm")
    print(f"all_success={result['all_success']} wall={wall:.1f}s -> {args.run / 'result.json'}")


if __name__ == "__main__":
    main()
