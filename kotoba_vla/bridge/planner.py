"""日本語の依頼を、VLA が理解できる正準スキルの列へ「翻訳」する LLM プランナ。

- LLM: Qwen3-8B（MLX 4bit、Mac 上でローカル推論）
- LLM が書けるのはカタログにあるスキル ID だけ。出力は検証し、違反したら理由を添えて再生成させる。
  → VLA に渡る文字列は必ず学習分布内の正準文になる（言い回しの揺れが VLA に届かない）
- カタログで実行できない依頼は「できない」と返す（無理に近いスキルへ丸めない）

比較用に、カタログを見せずに英訳だけさせる `translate_direct`（ナイーブな翻訳）も持つ。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from ..catalog import ROOT, Skill, SkillCatalog
from .retriever import SkillRetriever

SYSTEM_PROMPT = """あなたは家庭用ロボットアームの作業計画AIです。
ロボットは下の「スキル一覧」にあるスキルしか実行できません。各スキルは視覚言語行動モデル(VLA)が実行し、VLAはスキルの英語の正準文しか理解できません。

ユーザーの依頼を、実行順に並べたスキルの列に分解してください。
- 使ってよいのはスキル一覧の id だけです。新しいスキルを作ったり、文を言い換えたりしてはいけません。
- 依頼に含まれる作業だけを、依頼の順番どおりに並べます。依頼に無い作業を付け足してはいけません。
  1 つのスキルが依頼の複数の動作をまとめて表すこともあります（例:「開けて入れる」が 1 スキル）。
- 依頼は音声認識の結果なので、同音異義語や誤変換を含むことがあります（例: ボール→ボウル、さら→皿）。
  シーンにある物体を手がかりに、もっとも自然な意味に読み替えてください。
- 一覧のスキルでは実現できない部分があれば、steps に入れずに unsupported に日本語で書きます。
- 依頼全体が実現できなければ steps は空にします。

JSON だけを出力してください:
{"steps": [{"skill": "<スキルid>", "why": "<そのスキルを選んだ理由を短い日本語で>"}],
 "unsupported": "<実行できない部分の説明。なければ空文字>",
 "reply": "<ロボットとしてユーザーに返す短い日本語の返事。これから行う作業を「〜しますね」の形で伝える>"}"""

TRANSLATE_PROMPT = """Translate the following Japanese request to a household robot arm into one concise English imperative instruction for the robot.
Output only the English instruction, nothing else."""


@dataclass
class PlanStep:
    skill: Skill
    why: str


@dataclass
class Plan:
    request: str
    steps: list[PlanStep]
    unsupported: str
    reply: str
    raw: str = ""
    attempts: int = 1
    latency_s: float = 0.0
    candidates: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "request": self.request,
            "steps": [{"skill_id": s.skill.id, "instruction": s.skill.instruction, "why": s.why} for s in self.steps],
            "unsupported": self.unsupported,
            "reply": self.reply,
            "attempts": self.attempts,
            "latency_s": round(self.latency_s, 2),
            "candidates": [{"skill_id": c, "score": round(sc, 3)} for c, sc in self.candidates],
        }


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("JSON が見つからない")
    return json.loads(text[start : end + 1])


class LocalLLM:
    """mlx-lm で Qwen3 を動かす薄いラッパ。思考モードは切ってそのまま答えさせる。"""

    def __init__(self, model_name: str = "mlx-community/Qwen3-8B-4bit") -> None:
        from mlx_lm import load

        self.model, self.tokenizer = load(model_name)

    def chat(self, system: str, user: str, max_tokens: int = 768) -> str:
        from mlx_lm import generate

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


class SkillPlanner:
    def __init__(self, catalog: SkillCatalog, retriever: SkillRetriever, llm: LocalLLM,
                 max_retries: int = 2, n_examples: int = 2) -> None:
        self.catalog = catalog
        self.retriever = retriever
        self.llm = llm
        self.max_retries = max_retries
        self.n_examples = n_examples

    def _scene_text(self, scene_id: str) -> str:
        path = ROOT / "data" / "scene_glossary.json"
        glossary = json.loads(path.read_text(encoding="utf-8")).get(scene_id) if path.exists() else None
        if not glossary:
            return ""
        return "シーン: " + glossary["description"] + "\n" + "\n".join("- " + o for o in glossary["objects"]) + "\n\n"

    def _skill_list(self, request: str, scene_id: str) -> tuple[str, list[tuple[str, float]]]:
        hits = self.retriever.search(request, k=50, scene_id=scene_id)
        lines = []
        for h in hits:
            ex = "; ".join(self.retriever.examples(h.skill, self.n_examples))
            lines.append(f"- id: {h.skill.id}\n  正準文: {h.skill.instruction}\n  言い換え例: {ex}")
        return "\n".join(lines), [(h.skill.id, h.score) for h in hits]

    def plan(self, request: str, scene_id: str) -> Plan:
        t0 = time.perf_counter()
        skill_text, candidates = self._skill_list(request, scene_id)
        allowed = {s.id for s in self.catalog.in_scene(scene_id)}
        user = f"{self._scene_text(scene_id)}スキル一覧（依頼との類似度が高い順）:\n{skill_text}\n\n依頼: {request}"
        feedback = ""
        raw = ""
        for attempt in range(1, self.max_retries + 2):
            raw = self.llm.chat(SYSTEM_PROMPT, user + feedback)
            try:
                data = extract_json(raw)
                steps = []
                for st in data.get("steps", []):
                    skill = self.catalog.resolve(str(st.get("skill", "")))
                    if skill is None or skill.id not in allowed:
                        raise ValueError(f"一覧に無いスキル: {st.get('skill')}")
                    steps.append(PlanStep(skill, str(st.get("why", ""))))
                return Plan(request, steps, str(data.get("unsupported", "")), str(data.get("reply", "")),
                            raw=raw, attempts=attempt, latency_s=time.perf_counter() - t0, candidates=candidates)
            except (ValueError, json.JSONDecodeError) as e:
                feedback = f"\n\n前回の出力は不正でした（{e}）。スキル一覧の id だけを使い、JSON だけを出力し直してください。"
        # 全部失敗したら検索 1 位だけを実行する（安全側の縮退）
        top = self.catalog.by_id[candidates[0][0]]
        return Plan(request, [PlanStep(top, "LLM の出力が不正だったため検索 1 位を採用")], "", "",
                    raw=raw, attempts=self.max_retries + 1, latency_s=time.perf_counter() - t0,
                    candidates=candidates)

    def translate_direct(self, request: str) -> str:
        """比較用：カタログを使わずに英訳しただけの指示文（ナイーブな方式）。"""
        out = self.llm.chat(TRANSLATE_PROMPT, request, max_tokens=80)
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().splitlines()[0]
        return out.strip().strip('"')
