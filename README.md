# Kotoba-VLA — 日本語の声で頼める VLA ロボット

Physical AI 応用編1 最終課題。

日本語の音声で頼んだ作業を、LIBERO（robosuite / MuJoCo）上のロボットアームが VLA モデル π0.5 で実行します。
VLA は学習時に見た言い回しから外れると急に失敗するので、依頼を **VLA が学習時に見た言葉（正準文）へ翻訳する「言語ブリッジ」** を間に挟みました。
複数の作業をまとめた依頼は LLM がスキルの列に分解し、1 つずつ実行してゴール達成を確かめます。

- デモ動画: （YouTube の URL を記入）
- レポート: `report/` で生成する PDF

<!-- RESULTS -->

## 出発点: PARC2026 で分かったこと

この課題は、VLA 開発コンペ PARC2026 に参加して得た知見と資産を土台にしています。

| 事実（PARC2026 の本番で測定） | 意味 |
|---|---|
| 指示文を小文字にしただけで、本番スコアが 0.179 → 0.382 | VLA は学習時の言い回し（全文小文字の 40 文）から外れると急に弱くなる |
| 「ramekin の上の器」を取る課題で、20 回中 19 回は別の器を運んだ | 言葉と物体の対応づけに失敗すると、別の作業を自信満々に実行する |
| 同じ重み・同じ入力でも、乱数を固定しないとスコアが 1.76 倍ぶれた | 比較は乱数を固定し、同じ条件をそろえて行う |

本課題ではさらに、日本語の依頼を LLM で英訳しただけの文が学習時の文と一致した割合を測り、**30 文中 0 文**でした。
翻訳しただけでは「VLA に通じる言葉」にならない、というのが出発点です。

## しくみ

```
 日本語の声 ──▶ 音声認識 ──▶ スキル検索 ──▶ LLM 計画 ──▶ VLA 実行 ──▶ 検証 ──┐
               Whisper      言い換え       Qwen3-8B      π0.5＋LoRA   BDDL ゴール │
               (MLX)        2,052 文       スキル ID      (PARC T1      ＋ 1mm 接触 │
                            (Qwen3-Emb)    だけを選ぶ     と同一重み)              │
                                                              ▲                    │
                                                              └─ 失敗: 姿勢を戻して再試行
```

| 段 | 中身 | 工夫 |
|---|---|---|
| 音声認識 | Whisper large-v3-turbo（mlx-whisper） | Mac 上でローカル推論（平均 0.3 秒） |
| スキル検索 | Qwen3-Embedding-0.6B で、各スキルの正準文と PARC で集めた LIBERO-plus の言い換え 2,052 文を検索キーにする | 「Activate the appliance used for cooking」のような遠回しな言い方でも近いスキルを引ける |
| LLM 計画 | Qwen3-8B（MLX 4bit）。検索順に並べたスキル一覧とシーンの物体の説明を見せ、**スキル ID だけ**を JSON で返させる | 出力を検証し、一覧に無いスキルなら理由を添えて再生成。VLA に渡る文字列は必ず正準文になる。実行できない依頼は断る |
| VLA 実行 | π0.5（公開 finetuned 版 ＋ PARC2026 で自作した LoRA を係数 0.1 で焼き込み） | PARC 最終提出 Track1 と **SHA-256 が一致**する重みを手元で再構成。乱数を固定し、同じ入力なら同じ軌道 |
| 検証 | LIBERO の BDDL ゴール述語で成否を判定。対象外の物体が 1mm 以上動いたかも測る（PARC の採点と同じ基準） | 次のスキルの前にアームを初期姿勢へ戻す。失敗したら戻してから再試行 |

3 つの処理は Python の版が両立しない（PARC2026 で実測）ため、別プロセスに分けて HTTP でつないでいます。ROS 2 でノードを分けるのと同じ考え方です。

| プロセス | Python | 主なライブラリ |
|---|---|---|
| AI（音声認識・検索・LLM・動画合成） | 3.13 | mlx-lm, mlx-whisper, sentence-transformers |
| VLA 推論サーバ | 3.13 | torch 2.8（MPS, fp32）, transformers 5.3.0, lerobot 0.5.2 |
| シミュレータ | 3.10 | robosuite 1.4.0, mujoco 3.7.0, hf-libero 0.1.4 |

## PARC2026 から持ち込んだもの

| 資産 | 使い方 |
|---|---|
| 自作 LoRA（LIBERO-plus で学習、rank 16） `adapters/lp_ft_007500/` | 公開 π0.5 に焼き込んで最終提出 T1 と同じ重みを再構成（`scripts/merge_lora.py`） |
| 推論経路（`kotoba_vla/vla/pi05_t1.py`） | 重みの値照合・正規化統計の検査・乱数固定・`n_action_steps=12` など、本番で 0 点を出した落とし穴への対策込み |
| 言い換えプール 2,052 文 | スキル検索のキー、および「英語の言い換え」条件の評価データ（`scripts/build_skill_catalog.py`） |
| 1mm 接触判定 | 対象外の物体を動かしていないかの安全指標 |
| 評価の作法 | 乱数固定による決定論的な比較、選抜（開発セット）と確証（未使用セット）の分離 |

## 再現手順

Apple Silicon の Mac（M4 Max / 48GB で確認）を想定しています。

```bash
# 1. 環境（AI: Python 3.13 / シミュレータ: Python 3.10 / VLA: Python 3.13）
python3.13 -m venv .venv && .venv/bin/pip install -r envs/requirements-ai.txt
python3.10 -m venv .venv-sim && .venv-sim/bin/pip install -r envs/requirements-sim.txt
python3.13 -m venv .venv-vla && .venv-vla/bin/pip install -r envs/requirements-vla.txt

# 2. 外部の取得物（LIBERO 本体とアセット、π0.5 の公開重み、lerobot 0.5.2、トークナイザ）
scripts/setup_third_party.sh <PARC2026 の作業ディレクトリ>

# 3. 自作 LoRA を焼き込んで PARC 最終提出 T1 の重みを再構成し、SHA-256 を照合
.venv-vla/bin/python scripts/merge_lora.py --base weights/pi05_ft --adapter adapters/lp_ft_007500 \
    --scale 0.1 --out weights/pi05_parc_t1
cp configs/pi05_parc_t1/*.json weights/pi05_parc_t1/
shasum -a 256 weights/pi05_parc_t1/model.safetensors   # 126d63db21429a1c... と一致すること

# 4. 音声で頼んで実行（文面を渡すと音声合成してから音声認識に通す）
scripts/run_demo.sh outputs/runs/demo1 "コンロの火をつけて、そのあとボウルをコンロの上に置いて"
```

評価と動画:

```bash
.venv/bin/python scripts/eval_planner.py --tag dev                     # 言語ブリッジの正解率（開発セット）
.venv/bin/python scripts/eval_planner.py --tag holdout --commands data/eval_commands_ja_holdout.json
.venv/bin/python scripts/make_vla_set.py                               # VLA 評価に渡す文の一覧
.venv-sim/bin/python scripts/eval_vla.py                               # 渡す文ごとの VLA 成功率
.venv-sim/bin/python scripts/eval_chain.py                             # 複数ステップ依頼の完遂率
.venv/bin/python scripts/make_video.py                                 # 提出動画
.venv/bin/python -m pytest                                             # テスト（モデル不要、数秒）
```

## ディレクトリ

```
kotoba_vla/
  catalog.py          スキルカタログ（正準文・ゴール述語・言い換え）
  bridge/             音声認識・スキル検索・LLM 計画
  vla/                π0.5 推論（PARC T1 と同じ経路）・HTTP サーバとクライアント
  sim/                LIBERO シーン・実行エンジン（ホーム復帰・再試行・1mm 判定）・記録
  video/              デモ動画の合成
scripts/              カタログ生成・LoRA 焼き込み・評価・動画・デモ
data/                 スキルカタログ、シーンの物体説明、日本語の評価用依頼文（開発 / 未使用）
adapters/             PARC2026 で学習した LoRA
configs/              π0.5 の推論設定（PARC T1 と同じ）
results/              評価結果（集計 JSON と生ログ）
tests/                pytest（CI で実行）
```

## ライセンスと謝辞

- LIBERO（MIT）、hf-libero、robosuite、MuJoCo（Apache-2.0）
- lerobot（Apache-2.0）、π0.5 の重み（Physical Intelligence / lerobot が公開。PaliGemma 由来の部分は Gemma の利用規約に従う）。`adapters/` の LoRA はこの重みの派生物で、同じ条件に従う
- Qwen3 / Qwen3-Embedding（Apache-2.0）、Whisper（MIT）
- 言い換え文は LIBERO-plus の公開データに由来する
