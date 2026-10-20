"""説明用のスライド区間（タイトル・背景・構成図・結果・まとめ）。ナレーション音声付き。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .compose import (ACCENT, BG, FPS, H, MUTED, NG, OK, PANEL, PANEL_EDGE, TEXT, W, WARN, Canvas, Segment,
                      SegmentWriter, ease, font, wav_duration)


def render_card(out: Path, draw: Callable[[Canvas, float], None], narration: Path | None,
                min_seconds: float = 4.0, tail: float = 0.8, lead: float = 0.4) -> Segment:
    """draw(canvas, t) を毎フレーム呼ぶ。長さはナレーションに合わせる。"""
    dur = max(min_seconds, (wav_duration(narration) + lead + tail) if narration else 0.0)
    w = SegmentWriter(out)
    for k in range(int(dur * FPS)):
        c = Canvas()
        draw(c, k / FPS)
        w.write(c.array())
    return w.close([(narration, lead)] if narration else [])


def fade(t: float, start: float, length: float = 0.5) -> float:
    return ease((t - start) / length)


def _blend(color, a: float):
    return tuple(int(BG[i] + (color[i] - BG[i]) * a) for i in range(3))


def footer(c: Canvas, text: str) -> None:
    c.text((W - 40, H - 46), text, size=20, color=MUTED, anchor="ra")


# ---------------------------------------------------------------------------- 各スライド
def title_card(c: Canvas, t: float) -> None:
    a = fade(t, 0.2, 0.8)
    c.text((W // 2, 330), "Kotoba-VLA", size=120, bold=True, color=_blend(TEXT, a), anchor="ma")
    c.text((W // 2, 500), "日本語の声で、VLA ロボットに頼む", size=56, bold=True, color=_blend(ACCENT, fade(t, 0.8)), anchor="ma")
    c.text((W // 2, 610), "PARC2026 で分かった「VLA は言い回しに弱い」を、LLM の “通訳” で解く", size=34,
           color=_blend(TEXT, fade(t, 1.4)), anchor="ma")
    b = fade(t, 2.0)
    c.text((W // 2, 780), "Physical AI 応用編1 最終課題", size=28, color=_blend(MUTED, b), anchor="ma")
    c.text((W // 2, 830), "シミュレータ: LIBERO（robosuite / MuJoCo）   VLA: π0.5   すべて Mac 上でローカル推論",
           size=26, color=_blend(MUTED, b), anchor="ma")


def background_card(c: Canvas, t: float, cues=(0.8, 3.2, 5.6, 8.2)) -> None:
    c.text((80, 70), "出発点 ― PARC2026（VLA 開発コンペ）で分かったこと", size=46, bold=True)
    c.text((80, 140), "π0.5 などの VLA を LIBERO で動かして競うコンペに参加し、本番で次の事実を測った", size=28, color=MUTED)
    rows = [
        (cues[0], "0.179 → 0.382", "指示文を小文字にしただけで、本番スコアが 2 倍になった",
         "VLA は学習時に見た言い回し（全文小文字の 40 文）から外れると急に弱くなる"),
        (cues[1], "19 / 20", "「ramekin の上の器」を取る課題で、別の器を運んだ回数",
         "言葉と物体の対応づけ（接地）に失敗すると、自信満々に違うことをする"),
        (cues[2], "0 / 30", "日本語の依頼を LLM で英訳した文が、学習時の文と一致した数",
         "（本課題で測定）翻訳しただけでは「VLA に通じる言葉」にならない"),
    ]
    y = 230
    for start, big, head, sub in rows:
        a = fade(t, start)
        if a <= 0:
            y += 230
            continue
        c.panel((80, y, W - 80, y + 200), fill=_blend(PANEL, a), edge=_blend(PANEL_EDGE, a))
        c.text((120, y + 58), big, size=64, bold=True, color=_blend(WARN, a))
        c.text((720, y + 48), head, size=32, bold=True, color=_blend(TEXT, a))
        c.text((720, y + 112), sub, size=26, color=_blend(MUTED, a))
        y += 230
    if t > cues[3]:
        c.text((W // 2, 960), "→ ロボットへの依頼を「VLA が学習時に見た言葉」に翻訳する層を作る", size=36, bold=True,
               color=_blend(ACCENT, fade(t, cues[3])), anchor="ma")


def architecture_card(c: Canvas, t: float, cues=(0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4)) -> None:
    c.text((80, 70), "システム構成 ― 言語ブリッジ ＋ VLA ＋ 検証", size=46, bold=True)
    c.text((80, 140), "3 つのプロセス（AI / シミュレータ / VLA）を HTTP でつなぐ。ROS 2 のノード分割と同じ考え方", size=28, color=MUTED)
    boxes = [
        ("音声認識", "Whisper large-v3-turbo", "日本語の声を\n文字にする", cues[0]),
        ("スキル検索", "Qwen3-Embedding", "PARC の言い換え\n2,052 文から探す", cues[1]),
        ("LLM 計画", "Qwen3-8B（4bit）", "スキル ID だけを選ぶ\n→ 必ず正準文になる", cues[2]),
        ("VLA 実行", "π0.5 ＋ 自作 LoRA", "PARC 最終提出 T1 と\nビット単位で同じ重み", cues[3]),
        ("検証", "BDDL ゴール述語", "ゴール達成の判定\n＋ 1mm 接触の判定", cues[4]),
    ]
    bw, gap, y0 = 320, 40, 300
    x = 80
    for i, (head, model, desc, start) in enumerate(boxes):
        a = fade(t, start)
        if a > 0:
            edge = _blend(ACCENT if i == 3 else PANEL_EDGE, a)
            c.panel((x, y0, x + bw, y0 + 330), fill=_blend(PANEL, a), edge=edge)
            c.text((x + bw // 2, y0 + 34), head, size=36, bold=True, color=_blend(TEXT, a), anchor="ma")
            c.text((x + bw // 2, y0 + 96), model, size=24, color=_blend(ACCENT, a), anchor="ma")
            for j, line in enumerate(desc.split("\n")):
                c.text((x + bw // 2, y0 + 170 + 44 * j), line, size=24, color=_blend(MUTED, a), anchor="ma")
            if i < len(boxes) - 1:
                c.text((x + bw + gap // 2, y0 + 140), "→", size=40, color=_blend(MUTED, a), anchor="ma")
        x += bw + gap
    a = fade(t, cues[5])
    if a > 0:
        c.d.line((80 + 4 * (bw + gap) + bw // 2, y0 + 330, 80 + 4 * (bw + gap) + bw // 2, y0 + 420), fill=_blend(NG, a), width=4)
        c.d.line((80 + 3 * (bw + gap) + bw // 2, y0 + 420, 80 + 4 * (bw + gap) + bw // 2, y0 + 420), fill=_blend(NG, a), width=4)
        c.d.line((80 + 3 * (bw + gap) + bw // 2, y0 + 330, 80 + 3 * (bw + gap) + bw // 2, y0 + 420), fill=_blend(NG, a), width=4)
        c.text((80 + 3 * (bw + gap) + bw + gap // 2, y0 + 436), "失敗したら姿勢を戻して再試行", size=26,
               color=_blend(NG, a), anchor="ma")
    a = fade(t, cues[6])
    if a > 0:
        c.panel((80, 830, W - 80, 990), fill=_blend(PANEL, a), edge=_blend(PANEL_EDGE, a))
        c.text((120, 860), "PARC2026 から持ち込んだもの", size=28, bold=True, color=_blend(WARN, a))
        c.text((120, 912), "推論コード（重みの値照合・乱数固定）／ 自作 LoRA ／ 言い換え 2,052 文 ／ 1mm 接触判定 ／"
                           " 「選抜と確証は別セット」の評価作法", size=25, color=_blend(TEXT, a))


def bar_chart(c: Canvas, box, title: str, items: list[tuple[str, float, tuple]], t: float, start: float,
              unit: str = "%", note: str = "") -> None:
    a = fade(t, start)
    if a <= 0:
        return
    x0, y0, x1, y1 = box
    c.panel(box, fill=_blend(PANEL, a), edge=_blend(PANEL_EDGE, a))
    c.text((x0 + 28, y0 + 22), title, size=30, bold=True, color=_blend(TEXT, a))
    if note:
        c.text((x0 + 28, y0 + 66), note, size=21, color=_blend(MUTED, a))
    label_w, bar_x = 380, x0 + 420
    bar_max = x1 - bar_x - 120
    y = y0 + 112
    for label, value, color in items:
        grow = ease((t - start - 0.3) / 0.8)
        c.text((x0 + 28, y + 6), label, size=25, color=_blend(TEXT, a))
        bw = int(bar_max * max(0.0, min(1.0, value)) * grow)
        c.d.rounded_rectangle((bar_x, y + 4, bar_x + max(bw, 2), y + 38), radius=8, fill=_blend(color, a))
        c.text((bar_x + bw + 14, y + 6), f"{value * 100:.0f}{unit}", size=26, bold=True, color=_blend(color, a))
        y += 58


def closing_card(c: Canvas, t: float) -> None:
    c.text((80, 70), "まとめ", size=50, bold=True)
    pts = [
        ("言い回しの壁を、LLM の通訳で越えた", "日本語・遠回しな英語の依頼を、VLA が学習時に見た正準文へ変換して成功率を回復"),
        ("1 つの依頼を、複数の VLA スキルの連続実行に分解", "スキル間で姿勢を戻し、BDDL のゴールで成否を確かめ、失敗は再試行"),
        ("PARC2026 の資産をそのまま再利用", "最終提出と同じ π0.5＋LoRA を SHA-256 一致で再現し、評価の作法も持ち込んだ"),
    ]
    y = 200
    for i, (head, sub) in enumerate(pts):
        a = fade(t, 0.5 + 1.2 * i)
        c.text((80, y), f"{'①②③'[i]} {head}", size=38, bold=True, color=_blend(TEXT, a))
        c.text((134, y + 62), sub, size=27, color=_blend(MUTED, a))
        y += 170
    a = fade(t, 4.4)
    c.text((80, 760), "次の一歩", size=32, bold=True, color=_blend(WARN, a))
    c.text((80, 816), "実機（SO-101 など）での検証 ／ スキルが無い依頼を学習データ収集に回すデータフライホイール ／"
                      " VLM による成否判定", size=27, color=_blend(TEXT, a))


# 結果スライド用（暗い背景向けの系列色。系列 1 = 青、系列 2 = オレンジ）
SERIES_BLUE, SERIES_ORANGE = (57, 135, 229), (217, 89, 38)


def grouped_bars(c: Canvas, box, title: str, groups: list[tuple[str, list[tuple[str, float, str, tuple]]]],
                 t: float, start: float, reference: tuple[str, float] | None = None, note: str = "") -> None:
    """groups: [(グループ名, [(系列名, 値, 値ラベル, 色), ...]), ...]。reference は破線の基準値。"""
    a = fade(t, start)
    if a <= 0:
        return
    x0, y0, x1, y1 = box
    c.panel(box, fill=_blend(PANEL, a), edge=_blend(PANEL_EDGE, a))
    c.text((x0 + 28, y0 + 22), title, size=30, bold=True, color=_blend(TEXT, a))
    if note:
        c.text((x0 + 28, y0 + 66), note, size=21, color=_blend(MUTED, a))
    bar_x, bar_max = x0 + 360, (x1 - x0) - 360 - 200
    grow = ease((t - start - 0.3) / 0.8)
    y = y0 + 116
    for gname, series in groups:
        c.text((x0 + 28, y + 22), gname, size=26, bold=True, color=_blend(TEXT, a))
        for sname, value, vlabel, color in series:
            bw = int(bar_max * max(0.0, min(1.0, value)) * grow)
            c.d.rounded_rectangle((bar_x, y, bar_x + max(bw, 3), y + 34), radius=6, fill=_blend(color, a))
            c.text((bar_x + bw + 14, y + 2), vlabel, size=25, bold=True, color=_blend(TEXT, a))
            y += 44
        y += 22
    if reference:
        rx = bar_x + int(bar_max * reference[1])
        for yy in range(y0 + 104, y - 10, 16):
            c.d.line((rx, yy, rx, yy + 8), fill=_blend(MUTED, a), width=2)
        c.text((rx, y - 8), reference[0], size=20, color=_blend(MUTED, a), anchor="ma")


def legend(c: Canvas, xy, items: list[tuple[str, tuple]], t: float, start: float) -> None:
    a = fade(t, start)
    x, y = xy
    for name, color in items:
        c.d.rounded_rectangle((x, y + 6, x + 28, y + 26), radius=4, fill=_blend(color, a))
        c.text((x + 38, y), name, size=24, color=_blend(TEXT, a))
        x += 60 + int(c.d.textlength(name, font=font(24)))
