"""言語ブリッジのテスト（LLM・埋め込みモデルは偽物に差し替える。CI で数秒で回る）。"""

import json

import pytest

from kotoba_vla.bridge.planner import SkillPlanner, extract_json
from kotoba_vla.catalog import SkillCatalog


class FakeHit:
    def __init__(self, skill, score):
        self.skill, self.score, self.matched_text = skill, score, skill.instruction


class FakeRetriever:
    def __init__(self, catalog):
        self.catalog = catalog

    def search(self, query, k=5, scene_id=None):
        skills = [s for s in self.catalog.skills if scene_id is None or s.scene_id == scene_id]
        return [FakeHit(s, 0.5) for s in skills[:k]]

    def examples(self, skill, n):
        return list(skill.paraphrases[:n])


class ScriptedLLM:
    """決めた順に応答を返す偽 LLM。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def chat(self, system, user, max_tokens=768):
        self.prompts.append(user)
        return self.replies.pop(0)


@pytest.fixture(scope="module")
def catalog():
    return SkillCatalog.load()


def test_catalog_has_40_skills_and_goal_scene(catalog):
    assert len(catalog.skills) == 40
    goal = catalog.in_scene("libero_goal")
    assert len(goal) == 10
    assert sum(len(s.paraphrases) for s in catalog.skills) == 2052
    # 正準文はすべて小文字（VLA の学習分布に合わせる）
    assert all(s.instruction == s.instruction.lower() for s in catalog.skills)


def test_resolve_accepts_id_and_instruction(catalog):
    assert catalog.resolve("turn_on_the_stove").id == "turn_on_the_stove"
    assert catalog.resolve("Turn on the stove.").id == "turn_on_the_stove"
    assert catalog.resolve("switch on the cooker") is None


def test_extract_json_ignores_think_and_prose():
    raw = '<think>考え中</think>はい。{"steps": [], "unsupported": "x", "reply": "y"} 以上'
    assert extract_json(raw)["unsupported"] == "x"


def test_planner_returns_catalog_skills_in_order(catalog):
    llm = ScriptedLLM([json.dumps({
        "steps": [{"skill": "turn_on_the_stove", "why": "火"}, {"skill": "put_the_bowl_on_the_stove", "why": "器"}],
        "unsupported": "", "reply": "やりますね"})])
    plan = SkillPlanner(catalog, FakeRetriever(catalog), llm).plan("コンロをつけてボウルを置いて", "libero_goal")
    assert [s.skill.instruction for s in plan.steps] == ["turn on the stove", "put the bowl on the stove"]
    assert plan.attempts == 1


def test_planner_rejects_out_of_scene_skill_and_retries(catalog):
    bad = json.dumps({"steps": [{"skill": "pick_up_the_milk_and_place_it_in_the_basket"}], "unsupported": "", "reply": ""})
    good = json.dumps({"steps": [{"skill": "turn_on_the_stove"}], "unsupported": "", "reply": "はい"})
    llm = ScriptedLLM([bad, good])
    plan = SkillPlanner(catalog, FakeRetriever(catalog), llm).plan("火をつけて", "libero_goal")
    assert plan.attempts == 2
    assert "一覧に無いスキル" in llm.prompts[1]
    assert plan.steps[0].skill.id == "turn_on_the_stove"


def test_planner_keeps_refusal(catalog):
    llm = ScriptedLLM([json.dumps({"steps": [], "unsupported": "冷蔵庫は無い", "reply": "できません"})])
    plan = SkillPlanner(catalog, FakeRetriever(catalog), llm).plan("冷蔵庫を開けて", "libero_goal")
    assert plan.steps == [] and plan.unsupported


def test_planner_falls_back_to_top_retrieval_when_llm_is_broken(catalog):
    llm = ScriptedLLM(["壊れた出力"] * 3)
    plan = SkillPlanner(catalog, FakeRetriever(catalog), llm, max_retries=2).plan("何か", "libero_goal")
    assert len(plan.steps) == 1 and plan.attempts == 3
