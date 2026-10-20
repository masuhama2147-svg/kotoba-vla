"""音声指示 → 文字起こし → スキル検索 → LLM 計画 → plan.json（AI 側の環境で実行）。

  python scripts/plan_command.py --run outputs/runs/demo1 \
      --text "クリームチーズをボウルに入れてから、ボウルをお皿にのせて"

--text を渡すと macOS の音声合成で「人の声の指示」を作り、それを Whisper に聞かせる
（文字列を直接 LLM に渡すのではなく、必ず音声認識を通す）。--wav で録音済みの音声も使える。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kotoba_vla.bridge import speech  # noqa: E402
from kotoba_vla.bridge.planner import LocalLLM, SkillPlanner  # noqa: E402
from kotoba_vla.bridge.retriever import SkillRetriever  # noqa: E402
from kotoba_vla.catalog import SkillCatalog  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--text", help="指示の文面（音声合成してから音声認識に通す）")
    ap.add_argument("--wav", type=Path, help="録音済みの指示音声")
    ap.add_argument("--scene", default="libero_goal")
    ap.add_argument("--user-voice", default="Kyoko")
    ap.add_argument("--robot-voice", default="Eddy (日本語（日本）)")
    args = ap.parse_args()
    run = args.run
    (run / "audio").mkdir(parents=True, exist_ok=True)

    if args.wav:
        wav = args.wav
    else:
        wav = speech.synthesize(args.text, run / "audio" / "request.wav", voice=args.user_voice)

    t0 = time.perf_counter()
    transcript = speech.transcribe(wav)
    asr_s = time.perf_counter() - t0
    print(f"[ASR {asr_s:.2f}s] {transcript}")

    catalog = SkillCatalog.load()
    retriever = SkillRetriever(catalog)
    planner = SkillPlanner(catalog, retriever, LocalLLM())
    plan = planner.plan(transcript, args.scene)
    print(f"[LLM {plan.latency_s:.2f}s] {[s.skill.instruction for s in plan.steps]} / {plan.reply}")
    if plan.unsupported:
        print("  unsupported:", plan.unsupported)

    reply = plan.reply or ("ごめんなさい、それはできません。" if not plan.steps else "わかりました。")
    speech.synthesize(reply, run / "audio" / "reply.wav", voice=args.robot_voice)
    speech.synthesize("すべて完了しました。", run / "audio" / "done.wav", voice=args.robot_voice)

    out = {
        "text": args.text,
        "transcript": transcript,
        "asr_latency_s": round(asr_s, 2),
        "scene": args.scene,
        "plan": plan.to_dict(),
        "llm_raw": plan.raw,
    }
    (run / "plan.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("->", run / "plan.json")


if __name__ == "__main__":
    main()
