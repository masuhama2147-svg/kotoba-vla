#!/bin/zsh
# 重みがそろった後の一連の実験とデモ収録をまとめて行う。
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-8000}
SIM=(env MAGICK_HOME=/opt/homebrew PYTHONPATH=.:third_party/hf_libero .venv-sim/bin/python)
mkdir -p results outputs

if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
  PYTHONPATH=.:third_party/lerobot052 nohup .venv-vla/bin/python -m kotoba_vla.vla.server \
    --model-dir weights/pi05_parc_t1 --hf-home weights/hf_cache --port "$PORT" > outputs/vla_server.log 2>&1 &
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do sleep 5; done
fi
curl -s "http://127.0.0.1:$PORT/health"; echo

# 1) 渡す言葉ごとの VLA 成功率 → デモ 2（比較）の収録
"${SIM[@]}" scripts/eval_vla.py --server "http://127.0.0.1:$PORT" | tee outputs/eval_vla.log
cp outputs/eval/vla_instructions.json outputs/eval/vla_results.jsonl outputs/eval/vla_summary.json results/
("${SIM[@]}" scripts/record_contrast.py --server "http://127.0.0.1:$PORT" | tee outputs/record_contrast.log) || echo "[run_all] 比較デモの収録をスキップ"
# 2) 複数ステップ依頼の完遂率
"${SIM[@]}" scripts/eval_chain.py --server "http://127.0.0.1:$PORT" | tee outputs/eval_chain.log
# 3) デモ収録（720p）
for run in demo1 refusal; do
  "${SIM[@]}" scripts/execute_plan.py --run outputs/runs/$run --server "http://127.0.0.1:$PORT" --init 0
done
curl -s "http://127.0.0.1:$PORT/health" > outputs/vla_server_stats.json

# 4) 結果を results/ に集約（版管理する）
cp outputs/eval/planner_summary_dev.json outputs/eval/planner_summary_holdout.json \
   outputs/eval/planner_rows_dev.json outputs/eval/planner_rows_holdout.json \
   outputs/eval/vla_instructions.json outputs/eval/vla_results.jsonl outputs/eval/vla_summary.json \
   outputs/eval/chain_results.json outputs/eval/chain_summary.json results/
.venv/bin/python - <<'PY'
import json
d = json.load(open("outputs/eval/planner_summary_dev.json"))
s = json.load(open("outputs/vla_server_stats.json"))
json.dump({"asr_s": d["asr_latency_mean_s"], "llm_s": d["text"]["llm_latency_mean_s"],
           "vla_chunk_s": round(s["inference_s_total"] / max(1, s["n_inference"]), 3),
           "vla_requests": s["act_calls"], "vla_inferences": s["n_inference"]},
          open("results/timing.json", "w"), indent=1)
print(open("results/timing.json").read())
PY
