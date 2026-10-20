"""提出用レポート（PDF）を results/ の数値から組み立てる。

  .venv/bin/python report/build_report.py --account <omnicampusアカウント名> --youtube <URL> --github <URL>

HTML を書き、Google Chrome のヘッドレス印刷で PDF にする。
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "report"))
from figures import chain_figure, vla_figure, wilson  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text())


CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "Hiragino Sans", "Hiragino Kaku Gothic ProN", sans-serif; color: #1b1b1a; font-size: 10pt;
       line-height: 1.65; }
h1 { font-size: 19pt; margin: 0 0 2mm; line-height: 1.3; }
.sub { color: #52514e; font-size: 10.5pt; margin-bottom: 4mm; }
.meta { font-size: 9pt; color: #52514e; border-top: 1px solid #d8d7d2; border-bottom: 1px solid #d8d7d2;
        padding: 2mm 0; margin-bottom: 5mm; }
h2 { font-size: 13pt; margin: 6mm 0 2mm; padding-bottom: 1mm; border-bottom: 2px solid #2a78d6; }
h3 { font-size: 10.5pt; margin: 4mm 0 1mm; }
p { margin: 1.2mm 0 2mm; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 3mm; font-size: 9pt; }
th, td { border-bottom: 1px solid #e4e3df; padding: 1.4mm 2mm; text-align: left; vertical-align: top; }
th { background: #f3f2ee; font-weight: 600; }
td.num, th.num { text-align: right; white-space: nowrap; }
.box { background: #f3f6fb; border-left: 4px solid #2a78d6; padding: 2.5mm 4mm; margin: 3mm 0; }
.kpis { display: flex; gap: 3mm; margin: 3mm 0; }
.kpi { flex: 1; border: 1px solid #d8d7d2; border-radius: 3mm; padding: 2.5mm 3mm; }
.kpi b { display: block; font-size: 17pt; color: #1c5cab; line-height: 1.2; }
.kpi span { font-size: 8.5pt; color: #52514e; }
figure { margin: 2mm 0 3mm; }
figure img { width: 100%; }
figcaption { font-size: 8.5pt; color: #52514e; }
code { font-family: Menlo, monospace; font-size: 8.5pt; background: #f3f2ee; padding: 0 1mm; }
pre { font-family: Menlo, monospace; font-size: 8pt; background: #f3f2ee; padding: 2.5mm 3mm; line-height: 1.45;
      white-space: pre-wrap; }
ul { margin: 1mm 0 2mm; padding-left: 5mm; }
li { margin: 0.6mm 0; }
.small { font-size: 8.5pt; color: #52514e; }
.pb { page-break-before: always; }
"""


def build(args) -> str:
    vla = load("vla_summary.json")
    p_dev = load("planner_summary_dev.json")
    p_ho = load("planner_summary_holdout.json")
    chain = load("chain_summary.json")
    repro = load("t1_reproduction.json")
    timing = load("timing.json")
    notes = json.loads((ROOT / "report" / "notes.json").read_text())

    fig_dir = ROOT / "report" / "figures"
    fig_dir.mkdir(exist_ok=True)
    vla_figure(vla, fig_dir / "vla.png")
    chain_figure(chain, fig_dir / "chain.png")

    def vrow(key, label):
        s = vla[key]
        lo, hi = wilson(s["success"], s["n"])
        return (f"<tr><td>{label}</td><td class='num'>{s['success']}/{s['n']}</td><td class='num'><b>{pct(s['success_rate'])}</b>"
                f"</td><td class='num'>{pct(lo)}–{pct(hi)}</td><td class='num'>{pct(s['clean_rate'])}</td>"
                f"<td class='num'>{s.get('fail_did_other_task', 0)}</td></tr>")

    def prow(label, s):
        return (f"<tr><td>{label}</td><td class='num'>{s['n_single']}</td>"
                f"<td class='num'>{pct(s['retrieval_canon_acc'])}</td><td class='num'>{pct(s['retrieval_para_acc'])}</td>"
                f"<td class='num'><b>{pct(s['llm_single_acc'])}</b></td><td class='num'>{pct(s['llm_multi_exact'])}</td>"
                f"<td class='num'>{pct(s['llm_refusal'])}</td><td class='num'>{pct(s['direct_in_distribution'])}</td></tr>")

    ja_raw, ja_br = vla["ja_direct_translate"], vla["ja_kotoba_bridge"]
    canon = vla["canonical"]
    f, n = chain["full"], chain["naive"]
    sections = []
    sections.append(f"""
<h1>Kotoba-VLA：日本語の声で頼める VLA ロボット</h1>
<div class="sub">PARC2026 で分かった「VLA は言い回しに弱い」を、LLM による“言葉の通訳”で解く</div>
<div class="meta">Physical AI 応用編1 最終課題 ／ omnicampus アカウント：{html.escape(args.account)}<br>
動画：{html.escape(args.youtube)} ／ コード：{html.escape(args.github)}</div>

<h2>1. 概要</h2>
<p>日本語の音声で頼んだ作業を、シミュレータ LIBERO（robosuite / MuJoCo）上のロボットアームが VLA（視覚言語行動モデル）π0.5 で実行するシステムを作った。
VLA は学習時に見た言い回しから外れると急に失敗するため、依頼を <b>VLA が学習時に見た文（正準文）へ翻訳する「言語ブリッジ」</b>を間に挟んだ。
言語ブリッジは音声認識（Whisper）・多言語埋め込み検索・LLM（Qwen3-8B）で構成し、すべて Mac 上でローカルに動く。
複数の作業をまとめた依頼は、LLM がスキルの列に分解し、1 つずつ実行してゴール達成を確かめる。</p>
<div class="kpis">
 <div class="kpi"><b>{pct(ja_raw['success_rate'])} → {pct(ja_br['success_rate'])}</b><span>日本語の依頼での VLA 成功率<br>（英訳しただけ → Kotoba-VLA、各 {ja_br['n']} 試行）</span></div>
 <div class="kpi"><b>{pct(p_ho['text']['llm_single_acc'])}</b><span>未使用の依頼文でのスキル選択の正解率<br>（文字入力、{p_ho['text']['n_single']} 件。音声入力 {pct(p_ho['asr']['llm_single_acc'])}）</span></div>
 <div class="kpi"><b>{f['chain_success']}/{f['n']}</b><span>複数ステップ依頼の完遂<br>（姿勢復帰＋再試行なしでは {n['chain_success']}/{n['n']}）</span></div>
</div>
""")
    sections.append(f"""
<h2>2. 目的と背景 ― PARC2026 から持ち込んだ問い</h2>
<p>筆者は VLA 開発コンペ PARC2026 に参加し、π0.5 を LIBERO 系のタスクで動かして本番評価を受けた。そこで次の事実を測っている。</p>
<table>
<tr><th>PARC2026 本番で測った事実</th><th>示唆</th></tr>
<tr><td>指示文を小文字にする 1 行の変更だけで、Track1 のスコアが 0.179 → 0.382</td><td>VLA は学習時の言い回し（全文小文字の 40 文）から外れると急に弱くなる</td></tr>
<tr><td>「ramekin の上の器」を取る課題で、20 回中 19 回は別の器を運んだ</td><td>言葉と物体の対応づけに失敗すると、別の作業を自信満々に実行する</td></tr>
<tr><td>同じ重みでも推論の乱数を固定しないとスコアが 1.76 倍ぶれた</td><td>比較は乱数を固定し、同じ条件で行う必要がある</td></tr>
</table>
<p>人が実際にロボットへ頼むときの言葉は、日本語で、遠回しで、複数の作業を含む。これを学習済みの VLA にそのまま（あるいは機械的に英訳して）渡すと性能が落ちるはずである。
実際、本課題で日本語の依頼を LLM で英訳したところ、学習時の文と一致したものは <b>{p_dev['text']['n_single']} 文中 {round(p_dev['text']['direct_in_distribution'] * p_dev['text']['n_single'])} 文</b>だった。
そこで本課題の目的を「VLA を再学習せずに、人の言葉を VLA に通じる言葉へ変換し、複数の作業を安全にやり切らせる」とした。</p>
""")
    sections.append(f"""
<h2>3. システム</h2>
<pre>日本語の声 → 音声認識 → スキル検索 → LLM 計画 → VLA 実行 → 検証 ─┐
             Whisper     言い換え     Qwen3-8B     π0.5＋LoRA   BDDL ゴール │
                         2,052 文     ID だけ選ぶ   (PARC T1)    ＋1mm 接触  │
                                                      ↑                     │
                                                      └─ 失敗: 姿勢を戻して再試行</pre>
<h3>3.1 言語ブリッジ（音声認識・検索・LLM 計画）</h3>
<ul>
<li><b>音声認識</b>：Whisper large-v3-turbo（mlx-whisper）。平均 {timing['asr_s']:.1f} 秒。</li>
<li><b>スキル検索</b>：各スキルの正準文と、PARC2026 で集めた LIBERO-plus の言い換え 2,052 文（40 スキル × 約 51 文）を検索キーにし、Qwen3-Embedding-0.6B で依頼に近いスキルを並べる。</li>
<li><b>LLM 計画</b>：Qwen3-8B（MLX 4bit）に、シーンにある物体の日本語の説明と、検索順のスキル一覧（ID・正準文・言い換え例）を見せ、<b>スキル ID の列だけ</b>を JSON で返させる。
出力は検証し、一覧に無い ID なら理由を添えて再生成させる。VLA に渡る文字列は必ず正準文になるため、言い回しの揺れが VLA に届かない。
実行できない依頼（例：「冷蔵庫から牛乳を出して」）は、近いスキルに丸めずに断る。平均 {timing['llm_s']:.1f} 秒。</li>
</ul>
<h3>3.2 VLA 実行 ― PARC2026 最終提出モデルの再構成</h3>
<p>VLA には、PARC2026 本選 Track1 の最終提出と同じ重みを使った。公開されている π0.5（<code>lerobot/pi05_libero_finetuned_v044</code>）に、
PARC2026 で LIBERO-plus を使って自作した LoRA（rank 16、38 モジュール）を係数 0.1 で焼き込んだもので、手元で再構成した重みの SHA-256 は
提出物の記録 <code>{repro['expected_sha256'][:16]}…</code> と{'<b>一致した</b>' if repro['match'] else '一致しなかった'}。
推論経路も提出時のもの（画像 256px、LIBERO の観測前処理、<code>n_action_steps=12</code>）をそのまま使い、PARC で 0 点を出した落とし穴
（重みが黙って読み込まれない、正規化統計が空で素通しになる）を起動時に値で検査する。Apple GPU（MPS）では fp32 で動かし、12 行動ぶんの推論が {timing['vla_chunk_s']:.2f} 秒。</p>
<h3>3.3 検証と回復</h3>
<p>各スキルの成否は LIBERO の BDDL ゴール述語（例：<code>On(bowl, plate)</code>）で毎ステップ判定する。LIBERO-Goal の 10 スキルは物体と配置が共通でゴールだけが異なるので、
ゴール述語を差し替えれば同じシーンで複数のスキルを連続実行・判定できる。安全指標として、PARC の採点と同じく「対象外の物体が 1mm 以上動いたか」も測る。
VLA は初期姿勢から学習しているため、次のスキルの前にアームを初期姿勢へ戻し（OSC の差分指令による P 制御）、失敗したら戻してから 1 回だけ再試行する。</p>
<h3>3.4 プロセス構成</h3>
<p>VLA（transformers 5.3 / lerobot 0.5.2）とシミュレータ（Python 3.10 / robosuite 1.4 / numpy 1.26）は同じ Python 環境に同居できない（PARC で実測）。
そこで AI・VLA・シミュレータの 3 プロセスに分け、観測と行動を msgpack で HTTP 越しに受け渡した。ROS 2 でノードを分けるのと同じ考え方である。</p>
""")
    sections.append(f"""
<h2 class="pb">4. 実験</h2>
<h3>4.1 言語ブリッジの正解率</h3>
<p>LIBERO-Goal の 10 スキルについて、日本語の依頼文を開発セット（1 スキル 3 通り＋複数ステップ 6 件＋実行不可 3 件）と、
プロンプト調整の後に新しく書いた未使用セット（1 スキル 2 通り＋複数ステップ 4 件＋実行不可 2 件）に分けた（PARC で学んだ「選抜と確証は別のセットで」）。
音声入力は、macOS の音声合成で読み上げた音声を Whisper に通した結果を使った。</p>
<table>
<tr><th>セット・入力</th><th class="num">件数</th><th class="num">検索のみ<br>（正準文）</th><th class="num">検索のみ<br>（＋言い換え）</th><th class="num">LLM 計画</th><th class="num">複数ステップ<br>完全一致</th><th class="num">実行不可を<br>断れた</th><th class="num">英訳が<br>正準文と一致</th></tr>
{prow('開発・文字', p_dev['text'])}{prow('開発・音声', p_dev['asr'])}{prow('未使用・文字', p_ho['text'])}{prow('未使用・音声', p_ho['asr'])}
</table>
<p>{notes['planner']}</p>

<h3>4.2 渡す言葉による VLA の成功率</h3>
<p>同じスキル・同じ初期状態（各スキル 3 通り）・同じ乱数で、VLA に渡す文だけを変えた。乱数を固定しているので、同じ文・同じ初期状態なら結果も同じになる（決定論）。
英語の言い換えを言語ブリッジに通す条件では、評価に使う言い換え文を検索キーと LLM への例文から外した（答えを見ながら検索しないため）。</p>
<figure><img src="{img(fig_dir / 'vla.png')}"><figcaption>図 1　VLA（π0.5）の成功率。細線は 95% 信頼区間（Wilson）、破線は学習時の正準文を渡したときの成功率。</figcaption></figure>
<table>
<tr><th>VLA に渡した文</th><th class="num">成功</th><th class="num">成功率</th><th class="num">95% 区間</th><th class="num">1mm 基準<br>でも成功</th><th class="num">失敗のうち別の<br>作業をした</th></tr>
{vrow('canonical', '学習時の正準文（基準）')}{vrow('libero_plus_paraphrase', '英語の言い換えをそのまま')}{vrow('plus_kotoba_bridge', '英語の言い換え → Kotoba-VLA')}
{vrow('ja_direct_translate', '日本語 → LLM で英訳しただけ')}{vrow('ja_kotoba_bridge', '日本語 → Kotoba-VLA')}
</table>
<p>{notes['vla']}</p>

<h3>4.3 複数ステップの依頼</h3>
<p>2〜3 個の作業を含む日本語の依頼 6 件を、それぞれ 3 つの初期状態で実行した（計 {f['n']} 試行）。比較対象は、スキルの間で姿勢を戻さず、失敗しても再試行しない素朴な連続実行である。</p>
<figure><img src="{img(fig_dir / 'chain.png')}"><figcaption>図 2　複数ステップ依頼の完遂率（全スキルのゴール達成）。</figcaption></figure>
<p>{notes['chain']}</p>
""")
    sections.append(f"""
<h2>5. 工夫した点</h2>
<ul>
{''.join(f'<li>{x}</li>' for x in notes['ideas'])}
</ul>
<h2>6. 考察・限界・今後</h2>
<ul>
{''.join(f'<li>{x}</li>' for x in notes['discussion'])}
</ul>
<h2>7. 講義との対応</h2>
<table>
<tr><th>講義</th><th>本課題での対応</th></tr>
<tr><td>第1・2回 MLOps / RobOps</td><td>評価条件の固定（乱数・初期状態）、開発セットと未使用セットの分離、pytest による検査、結果 JSON の版管理</td></tr>
<tr><td>第3回 計算機の底</td><td>MPS での fp32 推論（約 14GB）と bf16 の不具合回避、HTTP＋msgpack による観測（画像 2 枚＋状態）の受け渡し、各段の遅延計測</td></tr>
<tr><td>第4・5回 ROS 2・LeRobot</td><td>処理をプロセス（ノード）に分けて通信でつなぐ構成、LeRobot の π0.5 推論経路と LoRA</td></tr>
<tr><td>第6回 オープンソース</td><td>LIBERO・robosuite・MuJoCo・lerobot・Qwen・Whisper を組み合わせ、ライセンスを README に明記</td></tr>
</table>
<h2>8. 再現方法と提出物</h2>
<p>コード一式は GitHub（{html.escape(args.github)}）にある。環境構築・重みの再構成（SHA-256 照合）・評価・動画生成の手順は README にまとめた。
評価の生ログと集計は <code>results/</code> に置いた。動画の実行シーンは 2 倍速で再生している（動画内に明記）。</p>
<p class="small">使用した主な OSS：LIBERO / hf-libero（MIT）、robosuite・MuJoCo（Apache-2.0）、lerobot（Apache-2.0）、π0.5 の重み（Physical Intelligence / lerobot。PaliGemma 由来部分は Gemma 利用規約）、
Qwen3・Qwen3-Embedding（Apache-2.0）、Whisper（MIT）。言い換え文は LIBERO-plus の公開データに由来する。</p>
""")
    return f"<!doctype html><html lang='ja'><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(sections)}</body></html>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", default="ACCOUNT")
    ap.add_argument("--youtube", default="（YouTube の URL）")
    ap.add_argument("--github", default="（GitHub の URL）")
    args = ap.parse_args()
    out_html = ROOT / "report" / "report.html"
    out_html.write_text(build(args), encoding="utf-8")
    pdf = ROOT / "report" / f"説明資料_PAI最終課題_{args.account}.pdf"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    out_html.as_uri()], check=True, capture_output=True)
    print(pdf)


if __name__ == "__main__":
    main()
