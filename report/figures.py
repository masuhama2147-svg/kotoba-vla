"""レポート用の図（matplotlib、印刷向けの明るい配色）。

配色: 系列 1 = 青（Kotoba-VLA を通す）、系列 2 = オレンジ（そのまま渡す）。
成否の色（緑・赤）は系列には使わない。基準（学習時の正準文）は灰色の破線。
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _setup_font() -> None:
    import unicodedata

    for f in font_manager.findSystemFonts():
        name = unicodedata.normalize("NFC", f)
        if "W3" in name and ("ヒラギノ角ゴシック" in name or "Hiragino" in name):
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            return


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def vla_figure(summary: dict, out: Path) -> None:
    """言葉の種類（英語の言い換え / 日本語）× 渡し方（そのまま / Kotoba-VLA）の成功率。"""
    _setup_font()
    groups = [("英語の言い換え\n（LIBERO-plus）", "libero_plus_paraphrase", "plus_kotoba_bridge"),
              ("日本語の依頼", "ja_direct_translate", "ja_kotoba_bridge")]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    h = 0.34
    for gi, (label, raw_key, bridge_key) in enumerate(groups):
        for j, (key, color, name) in enumerate(((raw_key, ORANGE, "そのまま渡す（日本語は英訳のみ）"),
                                                (bridge_key, BLUE, "Kotoba-VLA を通す"))):
            s = summary[key]
            y = gi + (j - 0.5) * (h + 0.04)
            ax.barh(y, s["success_rate"], height=h, color=color, label=name if gi == 0 else None)
            lo, hi = wilson(s["success"], s["n"])
            ax.plot([lo, hi], [y, y], color=INK2, linewidth=1.0)
            ax.text(min(hi, 1.0) + 0.015, y, f"{s['success_rate'] * 100:.0f}%（{s['success']}/{s['n']}）",
                    va="center", fontsize=8.5, color=INK)
    canon = summary["canonical"]
    ax.axvline(canon["success_rate"], color=INK2, linestyle=(0, (4, 3)), linewidth=1.0)
    ax.text(canon["success_rate"] + 0.01, len(groups) - 0.5, f"学習時の正準文\n{canon['success_rate'] * 100:.0f}%",
            ha="left", va="bottom", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(groups)), [g[0] for g in groups], fontsize=9, color=INK)
    ax.set_ylim(len(groups) - 0.45, -0.7)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100%"], fontsize=8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=8.5, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def chain_figure(summary: dict, out: Path) -> None:
    _setup_font()
    rows = [("そのまま連続実行", "naive", ORANGE), ("姿勢を戻す＋再試行\n（Kotoba-VLA）", "full", BLUE)]
    fig, ax = plt.subplots(figsize=(7.2, 1.7), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for i, (label, key, color) in enumerate(rows):
        s = summary[key]
        rate = s["chain_success"] / s["n"]
        ax.barh(i, rate, height=0.5, color=color)
        lo, hi = wilson(s["chain_success"], s["n"])
        ax.plot([lo, hi], [i, i], color=INK2, linewidth=1.0)
        ax.text(min(hi, 1.0) + 0.015, i, f"{rate * 100:.0f}%（{s['chain_success']}/{s['n']}）", va="center",
                fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9, color=INK)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100%"], fontsize=8)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
