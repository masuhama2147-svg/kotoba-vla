"""音声の入出力。

- 音声認識: Whisper large-v3-turbo（mlx-whisper、Mac 上でローカル推論）
- 音声合成: macOS の `say`（デモ動画で「人の声の指示」と「ロボットの返事」を作るため）

デモでは合成音声を指示として Whisper に聞かせる。マイク入力 (`record_microphone`) にも差し替えられる。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ASR_MODEL = "mlx-community/whisper-large-v3-turbo"


def synthesize(text: str, out_wav: str | Path, voice: str = "Kyoko", rate: int = 175) -> Path:
    """テキストを 16kHz モノラル wav にする（macOS 専用）。"""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "say.aiff"
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(out_wav)],
                       check=True)
    return out_wav


def transcribe(wav: str | Path, model: str = ASR_MODEL) -> str:
    import mlx_whisper

    result = mlx_whisper.transcribe(str(wav), path_or_hf_repo=model, language="ja")
    return result["text"].strip()


def record_microphone(out_wav: str | Path, seconds: float = 6.0) -> Path:
    """マイクから録音する（ffmpeg の avfoundation 入力を使う）。"""
    out_wav = Path(out_wav)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "avfoundation", "-i", ":0", "-t", str(seconds),
                    "-ar", "16000", "-ac", "1", str(out_wav)], check=True)
    return out_wav


def synthesize_sentences(sentences: list[str], out_wav: str | Path, voice: str = "Kyoko", rate: int = 185,
                         gap: float = 0.25) -> list[float]:
    """文ごとに音声合成して連結する。スライドの表示を文に合わせるため、各文の開始秒を返す。"""
    out_wav = Path(out_wav)
    parts, starts, t = [], [], 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for i, text in enumerate(sentences):
            w = synthesize(text, Path(tmp) / f"{i}.wav", voice=voice, rate=rate)
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                        str(w)], capture_output=True, text=True, check=True).stdout)
            starts.append(round(t, 2))
            parts.append(w)
            t += dur + gap
        silence = Path(tmp) / "gap.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                        "-t", str(gap), str(silence)], check=True)
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{w}'\nfile '{silence}'\n" for w in parts))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-ar", "16000", "-ac", "1", str(out_wav)], check=True)
    return starts
