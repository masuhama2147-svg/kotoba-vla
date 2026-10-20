"""言語ブリッジ（音声認識 → 検索 → LLM 計画）の正解率を測る。シミュレータは使わない（AI 側の環境）。

比較する方式:
  retrieval_canon : 正準文 40 本だけを検索キーにした検索の 1 位
  retrieval_para  : PARC の言い換え 2,052 文も検索キーに加えた検索の 1 位
  llm_planner     : 検索順に並べたスキル一覧を見せて LLM がスキル列を選ぶ（本システム）
  direct_translate: LLM に英訳だけさせた文（VLA に渡る文字列が正準文と一致するか）

入力は「文字のまま」と「音声合成 → Whisper で文字起こし」の 2 通り。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kotoba_vla.bridge import speech  # noqa: E402
from kotoba_vla.bridge.planner import LocalLLM, SkillPlanner  # noqa: E402
from kotoba_vla.bridge.retriever import SkillRetriever  # noqa: E402
from kotoba_vla.catalog import SkillCatalog  # noqa: E402

SCENE = "libero_goal"


def canon(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--commands", type=Path, default=ROOT / "data" / "eval_commands_ja.json")
    ap.add_argument("--tag", default="dev", help="出力ファイル名の接尾辞（dev / holdout）")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.commands.read_text())
    catalog = SkillCatalog.load()
    goal_ids = [s.id for s in catalog.in_scene(SCENE)]
    r_para = SkillRetriever(catalog, use_paraphrases=True)
    r_canon = SkillRetriever(catalog, use_paraphrases=False)
    llm = LocalLLM()
    planner = SkillPlanner(catalog, r_para, llm)

    # 入力（文字のまま / 音声認識を通したもの）
    items = []
    for sid, texts in data["single"].items():
        for k, t in enumerate(texts):
            items.append({"kind": "single", "skill": sid, "variant": k, "text": t, "expect": [sid]})
    for m in data["multi"]:
        items.append({"kind": "multi", "text": m["text"], "expect": m["expect"]})
    for t in data["unsupported"]:
        items.append({"kind": "unsupported", "text": t, "expect": []})

    if not args.no_asr:
        wav_dir = args.out / f"asr_wav_{args.tag}"
        for i, it in enumerate(items):
            wav = speech.synthesize(it["text"], wav_dir / f"{i:03d}.wav")
            t0 = time.perf_counter()
            it["asr"] = speech.transcribe(wav)
            it["asr_s"] = round(time.perf_counter() - t0, 2)
            print(f"[ASR] {it['text']} -> {it['asr']}", flush=True)

    rows = []
    for it in items:
        for source in ("text", "asr"):
            if source == "asr" and "asr" not in it:
                continue
            q = it["text"] if source == "text" else it["asr"]
            row = {**{k: it[k] for k in ("kind", "text", "expect")}, "source": source, "query": q}
            if it["kind"] == "single":
                row["skill"], row["variant"] = it["skill"], it["variant"]
                row["retrieval_canon"] = r_canon.search(q, k=1, scene_id=SCENE)[0].skill.id
                row["retrieval_para"] = r_para.search(q, k=1, scene_id=SCENE)[0].skill.id
            plan = planner.plan(q, SCENE)
            row["llm_planner"] = [s.skill.id for s in plan.steps]
            row["llm_unsupported"] = plan.unsupported
            row["llm_latency_s"] = round(plan.latency_s, 2)
            if it["kind"] != "unsupported":
                row["direct_translate"] = planner.translate_direct(q)
            rows.append(row)
            print(f"[{source}] {q} -> {row['llm_planner']} | {row.get('direct_translate', '')}", flush=True)

    (args.out / f"planner_rows_{args.tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    # 集計
    def acc(rs, key):
        return sum(r[key] == r["expect"][0] for r in rs) / max(1, len(rs))

    summary = {}
    for source in ("text", "asr"):
        rs = [r for r in rows if r["source"] == source]
        if not rs:
            continue
        single = [r for r in rs if r["kind"] == "single"]
        multi = [r for r in rs if r["kind"] == "multi"]
        uns = [r for r in rs if r["kind"] == "unsupported"]
        canon_set = {canon(catalog.by_id[s].instruction) for s in goal_ids}
        summary[source] = {
            "n_single": len(single),
            "retrieval_canon_acc": acc(single, "retrieval_canon"),
            "retrieval_para_acc": acc(single, "retrieval_para"),
            "llm_single_acc": sum(r["llm_planner"] == r["expect"] for r in single) / len(single),
            "direct_in_distribution": sum(canon(r["direct_translate"]) in canon_set for r in single) / len(single),
            "direct_exact_correct": sum(canon(r["direct_translate"]) == canon(catalog.by_id[r["skill"]].instruction)
                                        for r in single) / len(single),
            "n_multi": len(multi),
            "llm_multi_exact": sum(r["llm_planner"] == r["expect"] for r in multi) / max(1, len(multi)),
            "n_unsupported": len(uns),
            "llm_refusal": sum(len(r["llm_planner"]) == 0 for r in uns) / max(1, len(uns)),
            "llm_latency_mean_s": round(sum(r["llm_latency_s"] for r in rs) / len(rs), 2),
        }
    if any("asr_s" in it for it in items):
        summary["asr_latency_mean_s"] = round(sum(it["asr_s"] for it in items) / len(items), 2)
    (args.out / f"planner_summary_{args.tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))



if __name__ == "__main__":
    main()
