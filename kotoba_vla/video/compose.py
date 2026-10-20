"""デモ動画の合成（1920x1080 / 30fps）。

シミュレータが残した生データ（main.mp4, wrist.mp4, timeline.jsonl, plan.json, result.json, 音声）に
説明用のオーバーレイを重ね、区間ごとの mp4 を作ってから連結する。
"""

from __future__ import annotations

import json
import subprocess
import textwrap
import unicodedata
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30

# 配色（暗い背景にアクセント 1 色 + 成否の 2 色）
BG = (15, 20, 32)
PANEL = (27, 34, 50)
PANEL_EDGE = (52, 63, 88)
TEXT = (232, 236, 244)
MUTED = (150, 160, 182)
ACCENT = (96, 165, 250)
OK = (74, 222, 128)
NG = (248, 113, 113)
WARN = (251, 191, 36)


def _find_font(weight: str) -> str:
    for f in glob("/System/Library/Fonts/*.ttc"):
        name = unicodedata.normalize("NFC", Path(f).name)
        if name == f"ヒラギノ角ゴシック {weight}.ttc":
            return f
    for f in glob(str(Path.home() / "Library/Fonts/NotoSansCJK*")):
        return f
    raise FileNotFoundError("日本語フォントが見つからない")


_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = ("W6" if bold else "W3", size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(_find_font(key[0]), size)
    return _FONT_CACHE[key]


def wrap(text: str, width_chars: int) -> list[str]:
    """日本語混じりの文を、全角換算の文字数で折り返す。"""
    lines, cur, cur_w = [], "", 0.0
    for ch in text:
        w = 1.0 if unicodedata.east_asian_width(ch) in "WF" else 0.55
        if cur_w + w > width_chars and cur:
            lines.append(cur)
            cur, cur_w = "", 0.0
        cur += ch
        cur_w += w
    if cur:
        lines.append(cur)
    return lines


class Canvas:
    def __init__(self, bg=BG) -> None:
        self.img = Image.new("RGB", (W, H), bg)
        self.d = ImageDraw.Draw(self.img)

    def panel(self, box, title: str | None = None, edge=PANEL_EDGE, fill=PANEL, title_color=MUTED):
        self.d.rounded_rectangle(box, radius=14, fill=fill, outline=edge, width=2)
        if title:
            self.d.text((box[0] + 20, box[1] + 14), title, font=font(24, True), fill=title_color)

    def text(self, xy, s: str, size=28, color=TEXT, bold=False, anchor="la"):
        self.d.text(xy, s, font=font(size, bold), fill=color, anchor=anchor)

    def paragraph(self, xy, s: str, size=28, color=TEXT, width_chars=28, line_gap=1.45, bold=False) -> int:
        x, y = xy
        for line in wrap(s, width_chars):
            self.d.text((x, y), line, font=font(size, bold), fill=color)
            y += int(size * line_gap)
        return y

    def chip(self, xy, s: str, color, size=22, filled=True) -> int:
        f = font(size, True)
        x0, y0 = xy
        w = int(self.d.textlength(s, font=f)) + 24
        box = (x0, y0, x0 + w, y0 + size + 16)
        if filled:
            self.d.rounded_rectangle(box, radius=(size + 16) // 2, fill=color)
            self.d.text((x0 + 12, y0 + 7), s, font=f, fill=BG)
        else:
            self.d.rounded_rectangle(box, radius=(size + 16) // 2, outline=color, width=2)
            self.d.text((x0 + 12, y0 + 7), s, font=f, fill=color)
        return x0 + w

    def paste(self, arr: np.ndarray, xy, size=None, border=None):
        im = Image.fromarray(arr)
        if size:
            im = im.resize(size, Image.LANCZOS)
        self.img.paste(im, xy)
        if border:
            x, y = xy
            self.d.rectangle((x - 2, y - 2, x + im.width + 1, y + im.height + 1), outline=border, width=3)

    def array(self) -> np.ndarray:
        return np.asarray(self.img)


class VideoReader:
    """ffmpeg で mp4 を全フレーム読み込む（デモ 1 本は数百フレームなのでメモリに載る）。"""

    def __init__(self, path: Path, size: int) -> None:
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        self.frames = np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.frames[min(max(i, 0), len(self.frames) - 1)]


@dataclass
class Segment:
    """1 区間の映像と、その区間内に置く音声（区間先頭からの秒数）。"""

    path: Path
    audio: list[tuple[Path, float]] = field(default_factory=list)


class SegmentWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.n = 0
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
             str(path.with_suffix(".silent.mp4"))],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray, repeat: int = 1) -> None:
        b = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        for _ in range(repeat):
            self.proc.stdin.write(b)
            self.n += 1

    @property
    def t(self) -> float:
        return self.n / FPS

    def close(self, audio: list[tuple[Path, float]]) -> Segment:
        self.proc.stdin.close()
        self.proc.wait()
        mux_audio(self.path.with_suffix(".silent.mp4"), audio, self.path, self.n / FPS)
        self.path.with_suffix(".silent.mp4").unlink()
        return Segment(self.path, audio)


def wav_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def mux_audio(video: Path, audio: list[tuple[Path, float]], out: Path, duration: float) -> None:
    """無音の映像に、指定時刻から音声を重ねる（48kHz ステレオ AAC。区間の長さぴったりに揃える）。"""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
           "-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    filters, labels = [], ["[1:a]"]
    for i, (wav, at) in enumerate(audio):
        cmd += ["-i", str(wav)]
        ms = int(at * 1000)
        filters.append(f"[{i + 2}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={ms}|{ms}[a{i}]")
        labels.append(f"[a{i}]")
    filters.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0[aout]")
    cmd += ["-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "160k", "-t", f"{duration:.3f}", str(out)]
    subprocess.run(cmd, check=True)


def concat(segments: list[Path], out: Path) -> None:
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in segments))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    lst.unlink()


def load_run(run: Path) -> dict:
    timeline = [json.loads(line) for line in (run / "timeline.jsonl").read_text().splitlines() if line.strip()]
    return {
        "plan": json.loads((run / "plan.json").read_text()),
        "result": json.loads((run / "result.json").read_text()),
        "frames": [r for r in timeline if "event" not in r],
        "events": [r for r in timeline if "event" in r],
    }


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def fit_lines(text: str, width_chars: int) -> str:
    return "\n".join(textwrap.wrap(text, width_chars))
