"""VLA 評価の対応比較（同じスキル・同じ初期状態・同じ元の依頼で、渡す文だけが違う組）。

McNemar の正確検定（不一致の組だけを二項検定）で、そのまま渡す場合と言語ブリッジを通す場合を比べる。
失敗の内訳（別スキルのゴールを満たした＝言葉の取り違え）もここで数える。
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_vla import norm  # noqa: E402


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    res = {}
    for line in (ROOT / "results" / "vla_results.jsonl").read_text().splitlines():
        r = json.loads(line)
        res[(r["key_instruction"], r["goal_skill"], r["init"])] = r
    rows = json.loads((ROOT / "results" / "vla_instructions.json").read_text())
    out = {}
    for raw_c, br_c in (("libero_plus_paraphrase", "plus_kotoba_bridge"), ("ja_direct_translate", "ja_kotoba_bridge")):
        b = c = both = neither = 0
        other_raw = []
        for r in rows:
            if r["condition"] != raw_c:
                continue
            twin = next(x for x in rows if x["condition"] == br_c and x["goal_skill"] == r["goal_skill"]
                        and x["variant"] == r["variant"])
            for init in (0, 1, 2):
                a = res[(norm(r["instruction"]), r["goal_skill"], init)]
                z = res.get((norm(twin["instruction"]), r["goal_skill"], init), {"success": False})
                if a["success"] and not z["success"]:
                    b += 1
                elif z["success"] and not a["success"]:
                    c += 1
                elif a["success"]:
                    both += 1
                else:
                    neither += 1
                if not a["success"] and a.get("other_goals"):
                    other_raw.append({"goal": r["goal_skill"], "text": r["instruction"], "init": init,
                                      "did": a["other_goals"]})
        out[f"{raw_c}_vs_{br_c}"] = {"raw_only": b, "bridge_only": c, "both": both, "neither": neither,
                                     "mcnemar_p": round(mcnemar_exact(b, c), 5), "wrong_task_examples": other_raw}
    (ROOT / "results" / "vla_paired.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, v in out.items():
        print(k, {x: v[x] for x in ("raw_only", "bridge_only", "both", "neither", "mcnemar_p")})
        for e in v["wrong_task_examples"][:25]:
            print("   wrong-task:", e["goal"], "|", e["text"][:70], "| init", e["init"], "->", e["did"])


if __name__ == "__main__":
    main()
