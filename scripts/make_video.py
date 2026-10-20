"""提出用のデモ動画（1〜3 分、1920x1080）を組み立てる（AI 側の環境で実行）。

構成:
  タイトル → 出発点（PARC2026 の知見）→ システム構成 → デモ 1（複数ステップ）
  → デモ 2（英訳 vs Kotoba-VLA の比較）→ デモ 3（できない依頼を断る）→ 結果 → まとめ

  python scripts/make_video.py --out outputs/video/kotoba_vla_demo.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kotoba_vla.bridge.speech import synthesize_sentences  # noqa: E402
from kotoba_vla.catalog import SkillCatalog  # noqa: E402
from kotoba_vla.video import cards  # noqa: E402
from kotoba_vla.video.compose import Canvas, concat, wav_duration  # noqa: E402
from kotoba_vla.video.contrast_segment import ContrastRenderer  # noqa: E402
from kotoba_vla.video.demo_segment import DemoRenderer  # noqa: E402

NARRATOR = "Kyoko"

# ナレーション（文ごとに合成し、スライドの表示をその文の開始に合わせる）
NARRATION = {
    "title": ["コトバ ブイエルエー。", "日本語の声で、VLA ロボットに作業を頼めるようにしました。"],
    "background": [
        "出発点は、VLA の開発コンペ、パーク2026 です。",
        "本番では、指示文を小文字にしただけで、スコアが2倍になりました。",
        "言葉と物体の対応づけに失敗して、別の器を運ぶこともありました。",
        "日本語の依頼を LLM で英訳しても、学習時の文とは一つも一致しません。",
        "そこで、依頼を、VLA に通じる言葉へ翻訳する層をつくりました。",
    ],
    "architecture": [
        "声は、ウィスパーで文字にします。",
        "パークで集めた2052文の言い換えから、近いスキルを探し、",
        "LLM はスキルの ID だけを選ぶので、VLA には必ず学習時の文が届きます。",
        "VLA は、パークの最終提出と、ビット単位で同じ重みです。",
        "ゴール述語と、1ミリの接触判定で確かめ、",
        "失敗したら、姿勢を戻してやり直します。",
        "すべて、Mac の上で動きます。",
    ],
    "contrast": ["同じ依頼でも、VLA に渡す言葉で、結果が変わります。",
                 "左は英訳しただけの文、右はコトバ ブイエルエーが選んだ、学習時の文です。"],
    "closing": ["パークで得た知見と資産を、日本語で頼めるロボットとしてまとめました。", "ご覧いただき、ありがとうございました。"],
}
LEAD = 0.4  # render_card がナレーションを置く開始秒


def results_card_factory(vla: dict, planner: dict, chain: dict):
    B, O = cards.SERIES_BLUE, cards.SERIES_ORANGE

    def item(key, name, color):
        v = vla[key]
        label = f"{v['success_rate'] * 100:.0f}%（{v['success']}/{v['n']}）"
        fails = v["n"] - v["success"]
        if fails:
            label += f"  失敗 {fails} 件中 {v.get('fail_did_other_task', 0)} 件は別の作業"
        return (name, v["success_rate"], label, color)

    groups = [
        ("英語の言い換え", [item("libero_plus_paraphrase", "そのまま", O), item("plus_kotoba_bridge", "Kotoba", B)]),
        ("日本語の依頼", [item("ja_direct_translate", "英訳のみ", O), item("ja_kotoba_bridge", "Kotoba", B)]),
    ]
    canon = vla["canonical"]["success_rate"]
    f, n = chain["full"], chain["naive"]
    chain_groups = [("複数ステップ", [
        ("そのまま", n["chain_success"] / n["n"], f"{n['chain_success']}/{n['n']}", O),
        ("Kotoba", f["chain_success"] / f["n"], f"{f['chain_success']}/{f['n']}", B)])]
    ho = planner.get("holdout", {})

    def draw(c: Canvas, t: float) -> None:
        c.text((80, 56), "結果 ― LIBERO-Goal（10 スキル）で測定", size=44, bold=True)
        cards.legend(c, (1080, 66), [("そのまま渡す（日本語は英訳のみ）", O), ("Kotoba-VLA を通す", B)], t, 0.2)
        cards.grouped_bars(c, (80, 140, 1840, 560), "VLA（π0.5）の成功率 ― 渡す言葉だけを変える", groups, t, 0.3,
                           reference=(f"学習時の正準文 {canon * 100:.0f}%", canon),
                           note="同じスキル・同じ初期状態（3 通り）・同じ乱数で、渡す文だけを変えた", label_space=640)
        cards.grouped_bars(c, (80, 590, 1100, 1010), "複数ステップ依頼の完遂", chain_groups, t, 2.5,
                           note="姿勢を戻す＋失敗時に再試行（Kotoba）と、そのまま連続実行の比較")
        a = cards.fade(t, 4.0)
        if a > 0 and ho:
            c.panel((1140, 590, 1840, 1010))
            c.text((1170, 612), "スキル選択の正解率（未使用の依頼文）", size=28, bold=True)
            c.text((1170, 690), f"{ho['text']['llm_single_acc'] * 100:.0f}%", size=96, bold=True, color=B)
            c.text((1490, 720), f"文字入力 {ho['text']['n_single']} 件", size=24, color=cards.MUTED)
            c.text((1170, 840), f"音声入力でも {ho['asr']['llm_single_acc'] * 100:.0f}%。実行できない依頼は"
                                f" {ho['text']['llm_refusal'] * 100:.0f}% 断れた", size=24, color=cards.TEXT)
    return draw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "video" / "kotoba_vla_demo.mp4")
    ap.add_argument("--demo1", type=Path, default=ROOT / "outputs" / "runs" / "demo1")
    ap.add_argument("--contrast-left", type=Path, default=ROOT / "outputs" / "runs" / "contrast_direct")
    ap.add_argument("--contrast-right", type=Path, default=ROOT / "outputs" / "runs" / "contrast_bridge")
    ap.add_argument("--refusal", type=Path, default=ROOT / "outputs" / "runs" / "refusal")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--only", default="", help="カンマ区切りで一部の区間だけ作り直す")
    args = ap.parse_args()

    seg_dir = args.out.parent / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    nar_dir = args.out.parent / "narration"
    nar_dir.mkdir(parents=True, exist_ok=True)
    cues: dict[str, list[float]] = {}
    for key, sentences in NARRATION.items():
        cue_file = nar_dir / f"{key}.json"
        if not (nar_dir / f"{key}.wav").exists() or not cue_file.exists():
            starts = synthesize_sentences(sentences, nar_dir / f"{key}.wav", voice=NARRATOR, rate=215)
            cue_file.write_text(json.dumps(starts))
        cues[key] = [LEAD + x for x in json.loads(cue_file.read_text())]
    catalog = SkillCatalog.load()
    ev = ROOT / "results"
    only = set(args.only.split(",")) if args.only else None

    def want(name: str) -> bool:
        return only is None or name in only

    segs = []
    if want("title"):
        cards.render_card(seg_dir / "01_title.mp4", cards.title_card, nar_dir / "title.wav", min_seconds=5.0)
    if want("background"):
        cb = cues["background"]
        cards.render_card(seg_dir / "02_background.mp4",
                          lambda c, t: cards.background_card(c, t, cues=(cb[1], cb[2], cb[3], cb[4])),
                          nar_dir / "background.wav", lead=LEAD)
    if want("architecture"):
        ca = cues["architecture"]
        cards.render_card(seg_dir / "03_architecture.mp4",
                          lambda c, t: cards.architecture_card(c, t, cues=tuple(ca[:7])),
                          nar_dir / "architecture.wav", lead=LEAD)
    if want("demo1"):
        plan = json.loads((args.demo1 / "plan.json").read_text())
        caps = []
        if "ボール" in plan["transcript"]:
            caps += [{"phase": "asr", "text": "音声認識は「ボウル」を「ボール」と聞き取った"},
                     {"phase": "plan", "text": "それでも LLM はシーンの物体から bowl を選ぶ"}]
        caps += [{"phase": "search", "text": "PARC の言い換え 2,052 文から、近いスキルを検索"},
                 {"phase": "exec", "text": "VLA には学習時と同じ正準文だけを渡す"}]
        DemoRenderer(args.demo1, "デモ1 ｜ 2 つの作業をまとめて依頼", catalog, speed=args.speed,
                     captions=caps).render(seg_dir / "04_demo1.mp4")
    if want("contrast") and args.contrast_left.exists():
        req = json.loads((args.contrast_right / "plan.json").read_text())["transcript"]
        choice = json.loads((args.contrast_left / "result.json").read_text()).get("choice", {})
        cap = ""
        if choice.get("other_goals"):
            from kotoba_vla.video.labels import SKILL_JA

            did = SKILL_JA.get(choice["other_goals"][0], choice["other_goals"][0])
            cap = f"英訳した文では、頼んでいない「{did}」をしてしまった"
        ContrastRenderer(args.contrast_left, args.contrast_right, req, "デモ2 ｜ 渡す言葉で結果が変わる",
                         speed=args.speed).render(seg_dir / "05_contrast.mp4", nar_dir / "contrast.wav", caption=cap)
    if want("refusal") and args.refusal.exists():
        DemoRenderer(args.refusal, "デモ3 ｜ できない依頼は断る", catalog, speed=args.speed,
                     captions=[{"phase": "done", "text": "スキルに無い作業は、近いスキルへ丸めずに断る"}]
                     ).render(seg_dir / "06_refusal.mp4")
    if want("results") and (ev / "vla_summary.json").exists():
        vla = json.loads((ev / "vla_summary.json").read_text())
        planner = {t: json.loads((ev / f"planner_summary_{t}.json").read_text())
                   for t in ("dev", "holdout") if (ev / f"planner_summary_{t}.json").exists()}
        chain = json.loads((ev / "chain_summary.json").read_text()) if (ev / "chain_summary.json").exists() else {}
        synthesize_sentences(results_narration(vla, planner, chain), nar_dir / "results.wav", voice=NARRATOR, rate=215)
        cards.render_card(seg_dir / "07_results.mp4", results_card_factory(vla, planner, chain),
                          nar_dir / "results.wav", min_seconds=12.0)
    if want("closing"):
        cards.render_card(seg_dir / "08_closing.mp4", cards.closing_card, nar_dir / "closing.wav", min_seconds=7.0)

    segs = sorted(p for p in seg_dir.glob("0*.mp4") if not p.name.endswith(".silent.mp4"))
    concat(segs, args.out)
    total = wav_duration(args.out)
    print(f"{args.out}  {total:.1f} 秒  ({len(segs)} 区間)")


def results_narration(vla: dict, planner: dict, chain: dict) -> list[str]:
    def pct(k):
        return round(vla[k]["success_rate"] * 100)

    f, n = chain["full"], chain["naive"]
    return [
        "リベロ ゴールの10タスクで、VLA に渡す言葉だけを変えて比べました。",
        f"英語の遠回しな言い換えをそのまま渡すと、成功率は{pct('libero_plus_paraphrase')}パーセントに下がり、"
        "失敗の多くは、頼んでいない別の作業でした。",
        f"コトバ ブイエルエーを通すと{pct('plus_kotoba_bridge')}パーセントに戻り、別の作業をしてしまう失敗は、ゼロになりました。",
        f"日本語の依頼は、英訳だけでも{pct('ja_direct_translate')}パーセントと健闘しましたが、遠回しな依頼では別の作業をしてしまい、"
        f"コトバを通すと{pct('ja_kotoba_bridge')}パーセントでした。",
        f"複数の作業をまとめた依頼は、姿勢を戻して再試行することで、{n['n']}件中{n['chain_success']}件から"
        f"{f['chain_success']}件の完遂に増えました。",
    ]


if __name__ == "__main__":
    main()
