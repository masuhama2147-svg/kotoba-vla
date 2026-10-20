"""多言語埋め込みによるスキル検索。

検索キーは各スキルの正準指示文に加えて、PARC2026 で集めた LIBERO-plus の言い換え
（40 スキル × 約 51 文 = 2,052 文）。「コンロを点けて」のような日本語でも、
"Activate the appliance used for cooking" のような遠回しな英語でも、
同じスキルの言い換えのどれかに近ければ引き当てられる。

スキルのスコア = そのスキルに属する全キー（正準文 + 言い換え）とのコサイン類似度の最大値。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..catalog import ROOT, Skill, SkillCatalog

CACHE_DIR = ROOT / "outputs" / "cache"
QUERY_PROMPT = "Instruct: Given a household robot command, retrieve the robot task it asks for\nQuery: "


@dataclass
class Hit:
    skill: Skill
    score: float
    matched_text: str


class SkillRetriever:
    def __init__(self, catalog: SkillCatalog, model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                 device: str | None = None, use_paraphrases: bool = True,
                 exclude: set[str] | frozenset[str] = frozenset()) -> None:
        from sentence_transformers import SentenceTransformer

        self.catalog = catalog
        self.exclude = frozenset(exclude)
        self.model = SentenceTransformer(model_name, device=device)
        self.keys: list[str] = []
        self.key_skill: list[int] = []
        for i, s in enumerate(catalog.skills):
            # exclude: 評価で使う文を検索キーから外す（答えを見ながら検索しないため）
            texts = [s.instruction, *(p for p in (s.paraphrases if use_paraphrases else ()) if p not in exclude)]
            self.keys += texts
            self.key_skill += [i] * len(texts)
        self.key_skill_arr = np.array(self.key_skill)
        # 2,000 文の埋め込みは 30 秒ほどかかるので、モデル名とキー集合のハッシュでキャッシュする
        digest = hashlib.sha256((model_name + "\n" + "\n".join(self.keys)).encode()).hexdigest()[:16]
        cache = CACHE_DIR / f"emb_{digest}.npy"
        if cache.exists():
            self.key_emb = np.load(cache)
        else:
            self.key_emb = self.model.encode(self.keys, normalize_embeddings=True, batch_size=64,
                                             convert_to_numpy=True, show_progress_bar=False)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.key_emb)

    def examples(self, skill: Skill, n: int) -> list[str]:
        """プロンプトに載せる言い換え例（評価で除外した文は使わない）。"""
        return [p for p in skill.paraphrases if p not in self.exclude][:n]

    def search(self, query: str, k: int = 5, scene_id: str | None = None) -> list[Hit]:
        q = self.model.encode([QUERY_PROMPT + query], normalize_embeddings=True, convert_to_numpy=True)[0]
        # macOS の Accelerate 版 numpy は float32 の行列積で偽の浮動小数点警告を出すので float64 で計算する
        with np.errstate(all="ignore"):
            sims = self.key_emb.astype(np.float64) @ q.astype(np.float64)
        if not np.all(np.isfinite(sims)):
            raise FloatingPointError("類似度に NaN/inf が含まれる")
        best: dict[int, tuple[float, int]] = {}
        for key_i, (skill_i, sim) in enumerate(zip(self.key_skill, sims)):
            if scene_id is not None and self.catalog.skills[skill_i].scene_id != scene_id:
                continue
            if skill_i not in best or sim > best[skill_i][0]:
                best[skill_i] = (float(sim), key_i)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:k]
        return [Hit(self.catalog.skills[i], score, self.keys[key_i]) for i, (score, key_i) in ranked]
