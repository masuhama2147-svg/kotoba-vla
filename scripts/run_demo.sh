#!/bin/zsh
# 音声で頼んで、LIBERO 上の π0.5 に実行させるまでを 1 コマンドで行う。
#
#   scripts/run_demo.sh outputs/runs/my_demo                       # マイクで 6 秒録音して実行
#   scripts/run_demo.sh outputs/runs/my_demo "コンロの火をつけて"   # 文面を音声合成して実行
#
# 3 つの Python 環境を使い分ける（同じ環境に同居できないため。PARC2026 で実測）:
#   .venv      : 音声認識・検索・LLM 計画・動画合成（Python 3.13 / MLX）
#   .venv-vla  : π0.5 推論サーバ（transformers 5.3 / lerobot 0.5.2）
#   .venv-sim  : LIBERO シミュレータ（Python 3.10 / robosuite 1.4 / numpy 1.26）
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:?run ディレクトリを指定}
TEXT=${2:-}
INIT=${INIT:-0}
PORT=${PORT:-8000}

if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "[demo] VLA サーバを起動（重みの読み込みに 1〜2 分）"
  PYTHONPATH=.:third_party/lerobot052 nohup .venv-vla/bin/python -m kotoba_vla.vla.server \
    --model-dir weights/pi05_parc_t1 --hf-home weights/hf_cache --port "$PORT" > outputs/vla_server.log 2>&1 &
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do sleep 3; done
fi

mkdir -p "$RUN/audio"
if [ -n "$TEXT" ]; then
  .venv/bin/python scripts/plan_command.py --run "$RUN" --text "$TEXT"
else
  echo "[demo] 6 秒間、依頼を話してください"
  .venv/bin/python -c "from kotoba_vla.bridge.speech import record_microphone as r; r('$RUN/audio/request.wav', 6)"
  .venv/bin/python scripts/plan_command.py --run "$RUN" --wav "$RUN/audio/request.wav"
fi

MAGICK_HOME=/opt/homebrew PYTHONPATH=.:third_party/hf_libero .venv-sim/bin/python scripts/execute_plan.py \
  --run "$RUN" --server "http://127.0.0.1:$PORT" --init "$INIT"
