"""動画に表示するスキルの日本語名と、ゴール述語の表記。"""

SKILL_JA = {
    "open_the_middle_drawer_of_the_cabinet": "棚の真ん中の引き出しを開ける",
    "put_the_bowl_on_the_stove": "ボウルをコンロに置く",
    "put_the_wine_bottle_on_top_of_the_cabinet": "ワインボトルを棚の上に置く",
    "open_the_top_drawer_and_put_the_bowl_inside": "上の引き出しを開けてボウルを入れる",
    "put_the_bowl_on_top_of_the_cabinet": "ボウルを棚の上に置く",
    "push_the_plate_to_the_front_of_the_stove": "皿をコンロの手前へ押す",
    "put_the_cream_cheese_in_the_bowl": "クリームチーズをボウルに入れる",
    "turn_on_the_stove": "コンロの火をつける",
    "put_the_bowl_on_the_plate": "ボウルを皿にのせる",
    "put_the_wine_bottle_on_the_rack": "ワインボトルをラックに置く",
}


def goal_text(goal: list) -> str:
    return " ∧ ".join(f"{a[0]}({', '.join(a[1:])})" for a in goal)
