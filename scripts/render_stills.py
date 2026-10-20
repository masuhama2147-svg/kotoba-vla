"""動画のタイトル背景などに使う静止画をシミュレータから描く（シミュレータ側の環境で実行）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kotoba_vla.catalog import SkillCatalog  # noqa: E402
from kotoba_vla.sim.libero_scene import LiberoScene, setup_libero_config  # noqa: E402


def main() -> None:
    import imageio

    setup_libero_config(os.environ.get("LIBERO_ROOT", str(ROOT / "third_party" / "hf_libero")))
    out = ROOT / "outputs" / "video_assets"
    out.mkdir(parents=True, exist_ok=True)
    scene = LiberoScene(SkillCatalog.load().by_id["put_the_bowl_on_the_stove"])
    scene.reset(0)
    img = scene.inner.sim.render(camera_name="sideview", height=1080, width=1920)[::-1]
    imageio.imwrite(out / "title_bg.png", img)
    scene.close()
    print(out / "title_bg.png")


if __name__ == "__main__":
    main()
