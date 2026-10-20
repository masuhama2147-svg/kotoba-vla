"""1 回のデモ実行（run ディレクトリ）を、説明オーバーレイ付きの動画区間にする。

区間の流れ:
  聞き取り（指示音声）→ 計画（検索候補と LLM のスキル列、ロボットの返事）→ 実行（VLA、倍速）→ 完了
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .compose import (ACCENT, BG, FPS, MUTED, NG, OK, PANEL, TEXT, WARN, Canvas, Segment, SegmentWriter,
                      VideoReader, ease, font, load_run, wav_duration, wrap)
from .labels import SKILL_JA, goal_text

MAIN_XY, MAIN_SIZE = (40, 100), 900
RX, RW = 980, 900
BOX_ASR = (RX, 100, RX + RW, 292)
BOX_PLAN = (RX, 308, RX + RW, 690)
BOX_VLA = (RX, 706, RX + RW, 1000)
STAGES = ["音声認識  Whisper", "スキル検索  2,052文", "LLM計画  Qwen3-8B", "VLA実行  π0.5", "検証  BDDL＋1mm"]
SIM_HZ = 20


def _status_chip(c: Canvas, x_right: int, y: int, status: str) -> None:
    spec = {
        "pending": ("待機", MUTED, False),
        "running": ("▶ 実行中", ACCENT, True),
        "retry": ("▶ 再試行", WARN, True),
        "home": ("姿勢を戻す", MUTED, True),
        "ok": ("✓ 成功", OK, True),
        "ng": ("× 失敗", NG, True),
    }[status]
    f_w = int(c.d.textlength(spec[0], font=font(22, True)))
    c.chip((x_right - f_w - 24, y), spec[0], spec[1], size=22, filled=spec[2])


class DemoRenderer:
    def __init__(self, run: Path, header: str, catalog, speed: float = 2.0, video_size: int = 720,
                 captions: list[dict] | None = None, model_label: str = "π0.5（公開FT ＋ PARC自作LoRA×0.1）") -> None:
        self.run = Path(run)
        self.header = header
        self.catalog = catalog
        self.speed = speed
        self.data = load_run(self.run)
        self.main = VideoReader(self.run / "main.mp4", video_size)
        self.wrist = VideoReader(self.run / "wrist.mp4", 256)
        self.agent = VideoReader(self.run / "agent.mp4", 256)
        self.captions = captions or []
        self.model_label = model_label
        p = self.data["plan"]
        self.transcript = p["transcript"]
        self.steps = p["plan"]["steps"]
        self.candidates = p["plan"].get("candidates", [])[:3]
        self.unsupported = p["plan"].get("unsupported", "")
        self.reply = p["plan"].get("reply", "")
        self.verified = {e["skill_index"]: e for e in self.data["events"]}

    # ------------------------------------------------------------------ 共通の枠
    def base(self, main_frame: np.ndarray, stage: int, speed_badge: bool = False) -> Canvas:
        c = Canvas()
        c.text((40, 24), "Kotoba-VLA", size=34, bold=True, color=TEXT)
        c.text((300, 31), "日本語の声で、VLA ロボットに頼む", size=26, color=MUTED)
        c.text((1880, 28), self.header, size=28, bold=True, color=ACCENT, anchor="ra")
        c.paste(main_frame, MAIN_XY, (MAIN_SIZE, MAIN_SIZE))
        if speed_badge:
            c.chip((MAIN_XY[0] + 16, MAIN_XY[1] + 16), f"▶▶ {self.speed:g}倍速", WARN, size=24)
        # パイプラインの段階表示
        x = 40
        for i, s in enumerate(STAGES):
            color = ACCENT if i == stage else (OK if i < stage else MUTED)
            x = c.chip((x, 1022), s, color, size=22, filled=(i == stage)) + 14
            if i < len(STAGES) - 1:
                c.text((x - 4, 1026), "→", size=26, color=MUTED)
                x += 30
        return c

    def asr_box(self, c: Canvas, reveal: float, listening: bool) -> None:
        c.panel(BOX_ASR, "音声指示  →  文字起こし（Whisper large-v3-turbo・ローカル）")
        if listening:
            dots = "・" * (1 + int(reveal * 6) % 3)
            c.text((RX + 24, 172), f"聞き取り中{dots}", size=34, color=MUTED)
        else:
            n = max(1, int(len(self.transcript) * ease(reveal)))
            c.paragraph((RX + 24, 160), "「" + self.transcript[:n] + ("」" if n >= len(self.transcript) else ""),
                        size=32, width_chars=26, bold=True)

    def plan_box(self, c: Canvas, mode: str, reveal: float, statuses: list[str] | None = None) -> None:
        c.panel(BOX_PLAN, "LLM の計画  →  VLA に渡す正準指示（Qwen3-8B・ローカル）")
        y = BOX_PLAN[1] + 56
        if mode == "idle":
            c.text((RX + 24, y), "音声の文字起こしを待っています", size=24, color=MUTED)
            return
        if mode == "search":
            c.text((RX + 24, y), "PARC で集めた言い換え 2,052 文から近いスキルを検索", size=24, color=MUTED)
            y += 46
            for i, cand in enumerate(self.candidates):
                if reveal * 4 < i:
                    break
                sk = self.catalog.by_id[cand["skill_id"]]
                c.text((RX + 24, y), sk.instruction, size=27, color=TEXT)
                bar = int(260 * max(0.0, (cand["score"] - 0.4) / 0.5))
                c.d.rounded_rectangle((RX + 600, y + 6, RX + 600 + bar, y + 26), radius=6, fill=ACCENT)
                c.text((RX + 870, y), f"{cand['score']:.2f}", size=24, color=MUTED, anchor="ra")
                y += 48
            return
        if not self.steps:
            c.paragraph((RX + 24, y), "実行できるスキルがありません", size=32, color=WARN, bold=True)
            c.paragraph((RX + 24, y + 56), self.unsupported, size=26, color=TEXT, width_chars=30)
            return
        for i, st in enumerate(self.steps):
            if mode == "reveal" and reveal * (len(self.steps) + 0.5) < i + 0.5:
                break
            ja = SKILL_JA.get(st["skill_id"], st["instruction"])
            c.text((RX + 24, y), f"{'①②③④⑤'[i]} {ja}", size=30, bold=True)
            c.text((RX + 64, y + 44), f"VLA へ:  “{st['instruction']}”", size=25, color=ACCENT)
            if statuses:
                _status_chip(c, RX + RW - 20, y + 2, statuses[i])
            y += 98
        if self.unsupported:
            c.paragraph((RX + 24, y), "できない部分: " + self.unsupported, size=22, color=WARN, width_chars=38)

    def vla_box(self, c: Canvas, f: int, rec: dict | None) -> None:
        c.panel(BOX_VLA, "VLA の実行と検証")
        y0 = BOX_VLA[1] + 56
        c.paste(self.agent[f], (RX + 24, y0), (150, 150), border=PANEL)
        c.paste(self.wrist[f], (RX + 184, y0), (150, 150), border=PANEL)
        c.text((RX + 24, y0 + 160), "VLA への入力画像", size=18, color=MUTED)
        c.text((RX + 24, y0 + 186), "正面 ・ 手首", size=18, color=MUTED)
        x, y = RX + 360, BOX_VLA[1] + 56
        c.text((x, y), self.model_label, size=24, bold=True)
        if rec is None:
            c.text((x, y + 44), "待機中", size=24, color=MUTED)
            return
        idx = rec.get("skill_index", 0)
        skill = self.catalog.by_id[self.steps[idx]["skill_id"]]
        phase = rec.get("phase")
        if phase == "home":
            c.text((x, y + 44), "初期姿勢へ戻る（スクリプト制御）", size=23, color=MUTED)
        else:
            c.text((x, y + 44), f"入力文: {rec.get('vla_text', skill.instruction)}", size=23, color=ACCENT)
            c.text((x, y + 84), f"ステップ {rec.get('t', 0) + 1} / 300   試行 {rec.get('attempt', 1)}", size=23, color=TEXT)
        goal_ok = bool(rec.get("goal"))
        c.text((x, y + 128), "ゴール判定（BDDL）", size=20, color=MUTED)
        gt = goal_text(list(skill.goal))
        size = 22
        while size > 14 and c.d.textlength(gt, font=font(size)) > RX + RW - 24 - x:
            size -= 1
        c.text((x, y + 156), gt, size=size, color=OK if goal_ok else TEXT)
        d = float(rec.get("disturb_mm", 0.0))
        c.text((x, y + 196), f"対象外の物体の変位  {d:.1f} mm  （基準 1mm）", size=22, color=OK if d <= 1.0 else NG)

    def caption(self, c: Canvas, text: str, alpha: float = 1.0) -> None:
        if not text or alpha <= 0:
            return
        x0, y1 = MAIN_XY[0], MAIN_XY[1] + MAIN_SIZE
        lines = wrap(text, 30)
        h = 20 + 44 * len(lines)
        c.d.rectangle((x0, y1 - h, x0 + MAIN_SIZE, y1), fill=(10, 14, 24))
        for i, line in enumerate(lines):
            c.text((x0 + MAIN_SIZE // 2, y1 - h + 12 + 44 * i), line, size=30, bold=True, anchor="ma")

    def caption_at(self, phase: str, env_frame: int | None = None) -> str:
        for cap in self.captions:
            if cap.get("phase") != phase:
                continue
            if env_frame is None or cap.get("from", 0) <= env_frame <= cap.get("to", 10**9):
                return cap["text"]
        return ""

    # ------------------------------------------------------------------ 区間の書き出し
    def render(self, out: Path) -> Segment:
        w = SegmentWriter(out)
        audio: list[tuple[Path, float]] = []
        frame0 = self.main[0]
        frames = self.data["frames"]
        adir = self.run / "audio"

        # 1) 聞き取り
        req = adir / "request.wav"
        d_req = wav_duration(req) if req.exists() else 2.0
        if req.exists():
            audio.append((req, w.t + 0.3))
        n = int((d_req + 0.5) * FPS)
        for k in range(n):
            c = self.base(frame0, 0)
            self.asr_box(c, k / FPS, listening=True)
            self.plan_box(c, "idle", 0.0)
            self.vla_box(c, 0, None)
            self.caption(c, self.caption_at("listen"))
            w.write(c.array())
        for k in range(int(1.4 * FPS)):
            c = self.base(frame0, 0)
            self.asr_box(c, k / (0.9 * FPS), listening=False)
            self.plan_box(c, "idle", 0.0)
            self.vla_box(c, 0, None)
            self.caption(c, self.caption_at("asr"))
            w.write(c.array())

        # 2) 検索 → 計画（ロボットの返事を流す）
        for k in range(int(1.6 * FPS)):
            c = self.base(frame0, 1)
            self.asr_box(c, 1.0, listening=False)
            self.plan_box(c, "search", k / FPS)
            self.vla_box(c, 0, None)
            self.caption(c, self.caption_at("search"))
            w.write(c.array())
        rep = adir / "reply.wav"
        d_rep = wav_duration(rep) if rep.exists() else 2.0
        if rep.exists():
            audio.append((rep, w.t + 0.4))
        n = int(max(d_rep + 0.8, 3.0) * FPS)
        for k in range(n):
            c = self.base(frame0, 2)
            self.asr_box(c, 1.0, listening=False)
            self.plan_box(c, "reveal", k / (1.5 * FPS), ["pending"] * len(self.steps))
            self.vla_box(c, 0, None)
            self.caption(c, self.caption_at("plan"))
            w.write(c.array())

        # 3) 実行（倍速）
        if self.steps:
            per_out = self.speed * SIM_HZ / FPS
            n_out = int(len(frames) / per_out)
            last_banner = None
            for k in range(n_out):
                f = min(int(k * per_out), len(frames) - 1)
                rec = frames[f]
                statuses = self._statuses(f, rec)
                stage = 4 if rec.get("goal") else 3
                c = self.base(self.main[f], stage, speed_badge=True)
                self.asr_box(c, 1.0, listening=False)
                self.plan_box(c, "final", 1.0, statuses)
                self.vla_box(c, f, rec if rec.get("phase") in ("vla", "home") else None)
                ev = self._recent_event(f, per_out)
                if ev is not None:
                    last_banner = ev
                    self._banner(c, ev)
                else:
                    self.caption(c, self.caption_at("exec", f))
                w.write(c.array())

        # 4) 完了
        done = adir / "done.wav"
        res = self.data["result"]
        all_ok = bool(res.get("all_success"))
        if self.steps and all_ok and done.exists():
            audio.append((done, w.t + 0.2))
        last = self.main[len(self.main) - 1]
        for k in range(int(3.0 * FPS)):
            c = self.base(last, 4 if self.steps else 2)
            self.asr_box(c, 1.0, listening=False)
            self.plan_box(c, "final", 1.0, self._statuses(len(frames) - 1, frames[-1]) if self.steps else None)
            self.vla_box(c, len(self.wrist) - 1, None)
            n_ok = sum(o["success"] for o in res.get("outcomes", []))
            dmax = max([o["disturbance_mm"] for o in res.get("outcomes", [])] or [0.0])
            if self.steps:
                self.caption(c, self.caption_at("done") or f"{n_ok}/{len(self.steps)} スキル成功・対象外の物体の最大変位 {dmax:.1f}mm")
            else:
                self.caption(c, self.caption_at("done"))
            w.write(c.array())
        return w.close(audio)

    def _statuses(self, f: int, rec: dict) -> list[str]:
        out = []
        cur = rec.get("skill_index", 0)
        for i in range(len(self.steps)):
            ev = self.verified.get(i)
            if ev is not None and ev["frame"] <= f:
                out.append("ok" if ev.get("success") else "ng")
            elif i == cur and rec.get("phase") in ("vla", "home"):
                if rec.get("phase") == "home":
                    out.append("home")
                else:
                    out.append("retry" if rec.get("attempt", 1) > 1 else "running")
            else:
                out.append("pending")
        return out

    def _recent_event(self, f: int, per_out: float):
        window = int(1.6 * FPS * per_out)
        for ev in self.verified.values():
            if 0 <= f - ev["frame"] <= window:
                return ev
        return None

    def _banner(self, c: Canvas, ev: dict) -> None:
        i = ev["skill_index"]
        skill = self.catalog.by_id[self.steps[i]["skill_id"]]
        ok = ev.get("success")
        x0, y1 = MAIN_XY[0], MAIN_XY[1] + MAIN_SIZE
        c.d.rectangle((x0, y1 - 118, x0 + MAIN_SIZE, y1), fill=(10, 14, 24))
        head = f"{'✓' if ok else '×'} {'①②③④⑤'[i]} {SKILL_JA.get(skill.id, skill.instruction)} — {'達成' if ok else '未達'}"
        c.text((x0 + 24, y1 - 104), head, size=34, bold=True, color=OK if ok else NG)
        c.text((x0 + 24, y1 - 50), f"{goal_text(list(skill.goal))}   変位 {ev.get('disturb_mm', 0):.1f}mm", size=24, color=TEXT)
