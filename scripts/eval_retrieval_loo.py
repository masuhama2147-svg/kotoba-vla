"""言い換えプールが検索に効くかを、英語の言い換え文で Leave-one-out 評価する（AI 側の環境）。

LIBERO-Goal の各言い換え文をクエリにし、
  (a) 正準文 10 本だけを検索キーにした場合
  (b) 正準文＋言い換え（クエリ自身を除く）を検索キーにした場合
の 1 位の正解率を比べる。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kotoba_vla.bridge.retriever import QUERY_PROMPT, SkillRetriever  # noqa: E402
from kotoba_vla.catalog import SkillCatalog  # noqa: E402


def main() -> None:
    catalog = SkillCatalog.load()
    r = SkillRetriever(catalog)
    goal_idx = [i for i, s in enumerate(catalog.skills) if s.scene_id == "libero_goal"]
    keys = np.array(r.keys, dtype=object)
    owner = r.key_skill_arr
    in_goal = np.isin(owner, goal_idx)
    is_canon = np.array([k == catalog.skills[o].instruction for k, o in zip(r.keys, owner)])
    queries = [(p, i) for i in goal_idx for p in catalog.skills[i].paraphrases]
    q_emb = r.model.encode([QUERY_PROMPT + q for q, _ in queries], normalize_embeddings=True, batch_size=64,
                           convert_to_numpy=True, show_progress_bar=False).astype(np.float64)
    K = r.key_emb.astype(np.float64)
    hit_canon = hit_para = 0
    with np.errstate(all="ignore"):
        S = q_emb @ K.T
    for qi, (q, true_i) in enumerate(queries):
        sims = S[qi]
        m_c = in_goal & is_canon
        hit_canon += owner[m_c][np.argmax(sims[m_c])] == true_i
        m_p = in_goal & (keys != q)
        hit_para += owner[m_p][np.argmax(sims[m_p])] == true_i
    n = len(queries)
    out = {"n_queries": n, "top1_canonical_only": round(hit_canon / n, 3), "top1_with_paraphrases_loo": round(hit_para / n, 3)}
    (ROOT / "results" / "retrieval_loo.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
