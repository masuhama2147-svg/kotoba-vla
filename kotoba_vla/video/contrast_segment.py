"""同じ依頼・同じ初期状態で「英訳しただけの文」と「言語ブリッジの正準文」を並べて見せる区間。"""

from __future__ import annotations

import json
from pathlib import Path

from .compose import (ACCENT, FPS, MUTED, NG, OK, TEXT, WARN, Canvas, Segment, SegmentWriter, VideoReader,
                      wav_duration)

VIEW = 800
LEFT_X, RIGHT_X, VIEW_Y = 80, 1040, 210
SIM_HZ = 20


class ContrastRenderer:
    def __init__(self, left_run: Path, right_run: Path, request_ja: str, header: str, speed: float = 2.0,
                 video_size: int = 720) -> None:
        self.runs = []
        for run in (left_run, right_run):
            run = Path(run)
            frames = [json.loads(x) for x in (run / "timeline.jsonl").read_text().splitlines() if x.strip()]
            self.runs.append({
                "video": VideoReader(run / "main.mp4", video_size),
                "frames": [f for f in frames if "event" not in f],
                "result": json.loads((run / "result.json").read_text()),
                "plan": json.loads((run / "plan.json").read_text()),
            })
        self.request = request_ja
        self.header = header
        self.speed = speed

    def _frame(self, k_env: list[int], finished: list[bool], narration_cap: str = "") -> Canvas:
        c = Canvas()
        c.text((40, 24), "Kotoba-VLA", size=34, bold=True)
        c.text((1880, 28), self.header, size=28, bold=True, color=ACCENT, anchor="ra")
        c.text((960, 104), f"依頼「{self.request}」", size=38, bold=True, anchor="ma")
        c.text((960, 160), "同じ初期状態・同じ VLA（π0.5）。違うのは VLA に渡す言葉だけ", size=26, color=MUTED, anchor="ma")
        labels = [("英訳しただけ", NG), ("Kotoba-VLA（正準文に変換）", ACCENT)]
        for side, (x, run) in enumerate(zip((LEFT_X, RIGHT_X), self.runs)):
            f = min(k_env[side], len(run["video"]) - 1)
            c.paste(run["video"][f], (x, VIEW_Y), (VIEW, VIEW))
            head, color = labels[side]
            c.chip((x + 16, VIEW_Y + 16), head, color, size=26)
            text = run["plan"]["plan"]["steps"][0]["instruction"] if run["plan"]["plan"]["steps"] else "（計画なし）"
            c.d.rectangle((x, VIEW_Y + VIEW - 70, x + VIEW, VIEW_Y + VIEW), fill=(10, 14, 24))
            c.text((x + 20, VIEW_Y + VIEW - 52), f"VLA へ: “{text}”", size=26, color=TEXT)
            if finished[side]:
                ok = run["result"]["all_success"]
                c.chip((x + VIEW - 210, VIEW_Y + 16), "✓ ゴール達成" if ok else "× 達成できず", OK if ok else NG, size=28)
        c.chip((1800 - 210, 1030), f"▶▶ {self.speed:g}倍速", WARN, size=22)
        if narration_cap:
            c.text((960, 1036), narration_cap, size=26, color=MUTED, anchor="ma")
        return c

    def render(self, out: Path, narration: Path | None = None, caption: str = "") -> Segment:
        w = SegmentWriter(out)
        audio = [(narration, 0.3)] if narration else []
        intro = max(2.5, wav_duration(narration) * 0.5) if narration else 2.5
        for _ in range(int(intro * FPS)):
            w.write(self._frame([0, 0], [False, False], caption).array())
        per_out = self.speed * SIM_HZ / FPS
        lens = [len(r["frames"]) for r in self.runs]
        n_out = int(max(lens) / per_out) + 1
        for k in range(n_out):
            f = int(k * per_out)
            w.write(self._frame([f, f], [f >= lens[0] - 1, f >= lens[1] - 1], caption).array())
        last = self._frame(lens, [True, True], caption).array()
        w.write(last, int(2.5 * FPS))
        return w.close(audio)
