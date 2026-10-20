"""実行系の純粋な部品のテスト（シミュレータ・モデル無しで回る）。"""

import subprocess
import sys
from pathlib import Path

import numpy as np

from kotoba_vla.sim.runner import SkillOutcome
from kotoba_vla.vla.protocol import decode_arrays, encode_arrays

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_roundtrip_keeps_dtype_and_shape():
    obs = {
        "agentview_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, -0.2, 1.0]),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
    }
    back = decode_arrays(encode_arrays(obs))
    for k, v in obs.items():
        assert back[k].dtype == v.dtype and back[k].shape == v.shape
        np.testing.assert_array_equal(back[k], v)


def test_clean_success_uses_parc_1mm_rule():
    ok = SkillOutcome("s", "t", True, 1, 100, 0.8, "plate_1")
    touched = SkillOutcome("s", "t", True, 1, 100, 1.2, "plate_1")
    failed = SkillOutcome("s", "t", False, 2, 600, 0.0, "")
    assert ok.clean_success and not touched.clean_success and not failed.clean_success


def test_disturbance_ignores_objects_of_interest():
    from kotoba_vla.sim.libero_scene import LiberoScene

    start = {"bowl": np.zeros(3), "plate": np.zeros(3)}
    now = {"bowl": np.array([0.3, 0, 0]), "plate": np.array([0.0005, 0.0003, 0])}
    d, name = LiberoScene.max_disturbance(start, now, ignore={"bowl"})
    assert name == "plate" and abs(d - 0.0008) < 1e-9


def test_vla_server_normalization_matches_eval_cache_key():
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_vla import norm

    from kotoba_vla.vla.pi05_t1 import normalize_instruction

    for s in ["Turn on the stove.", "  Pick the bowl  and place it", "put the bowl on the plate"]:
        assert norm(s) == normalize_instruction(s)


def test_merge_lora_selftest():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "merge_lora.py"), "--selftest"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
