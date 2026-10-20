#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merge_lora.py — 公開 π0.5 に自作 LoRA を焼き込んで「完全チェックポイント」を作る。

PARC2026 本選で作った lora_merge.py の移植。最終提出 T1 は
  公開 finetuned π0.5（lerobot/pi05_libero_finetuned_v044）
  ＋ 自作 LoRA（LIBERO-plus で学習、rank16 / alpha8、7,500 step）を係数 0.1 で焼き込み
であり、このスクリプトで手元に同じ重みを再現できる（SHA-256 で照合）。

設計の要点（過去に踏んだ罠を塞ぐ）
--------------------------------
0. **同梱物は実体コピーで持ってくる（既定）。** 提出物にリンクを含められない評価系があったため。
   ディスクを節約したい測定用ビルドのときだけ `--link-mode hardlink` を明示する。
1. **正規化統計は配布側のものだけを使う。** lerobot は学習時にデータセット統計から
   normalizer を作り直すので、学習成果物の統計をそのまま持ってくるとモデルが壊れる
   （実測済み: 自作教師データの統計で v5 が 47.3% まで落ちた）。ここでは
   model.safetensors だけを書き換え、統計ファイル・config・vendor は
   **配布側からハードリンク**する。ハードリンクなのでディスクも食わない。
2. **キーが 1 つでも対応付かなければ落とす。** 黙って飛ばすと「LoRA を焼いたつもりで
   素の重みを提出する」ことになる。これは気付けない最悪の失敗。
3. **--scale で寄与の強さを連続的に変えられる。** W' = W + scale·(alpha/r)·B·A。
   scale=1.0 が PEFT の merge_and_unload と同一。0 < scale < 1 で弱められる。
   非劣性が確認できた最大の scale を採るのが安全。
4. **非対象テンソルのビット一致を検証する。** 触っていないことを証明できないと、
   「壊していない」と言えない。

使い方
------
    python3 scripts/merge_lora.py --base weights/pi05_ft --adapter adapters/lp_ft_007500 \
                                  --scale 0.1 --out weights/pi05_parc_t1
    python3 scripts/merge_lora.py --selftest        # 重み不要の自己検査
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------- キーの対応付け

_LORA_SUFFIX = re.compile(r"\.lora_(A|B)(?:\.default)?\.weight$")
_STRIP_PREFIXES = ("base_model.model.", "base_model.", "model.model.")


def adapter_key_to_module(key: str) -> tuple[str, str] | None:
    """adapter のテンソル名から (モジュールのパス, 'A' or 'B') を取り出す。

    PEFT は `base_model.model.<path>.lora_A.weight` の形で保存する。
    `.default` が挟まる版もあるので両方受ける。
    """
    m = _LORA_SUFFIX.search(key)
    if not m:
        return None
    path = key[: m.start()]
    for p in _STRIP_PREFIXES:
        if path.startswith(p):
            path = path[len(p):]
            break
    return path, m.group(1)


def candidate_base_keys(path: str) -> list[str]:
    """base ckpt 側の重みキーの候補を、確からしい順に返す。

    lerobot の保存形は前置きが 1 段ずれることがあるので、
    `<path>.weight` / `model.<path>.weight` / `<path>` を順に試す。
    """
    out = [f"{path}.weight", f"model.{path}.weight"]
    if path.startswith("model."):
        out.append(f"{path[len('model.'):]}.weight")
    out.append(path)
    seen, uniq = set(), []
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


# ---------------------------------------------------------------- 本体

def merge(base_dir: Path, adapter_dir: Path, out_dir: Path, scale: float,
          dry_run: bool = False, link_mode: str = "copy") -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg_p = adapter_dir / "adapter_config.json"
    if not cfg_p.is_file():
        raise SystemExit(f"adapter_config.json が無い: {cfg_p}")
    acfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    r = int(acfg.get("r") or 0)
    lora_alpha = float(acfg.get("lora_alpha") or 0)
    if r <= 0 or lora_alpha <= 0:
        raise SystemExit(f"adapter_config の r / lora_alpha が読めない: r={r} alpha={lora_alpha}")
    if acfg.get("use_dora"):
        raise SystemExit("DoRA は未対応（B@A の単純加算では表せない）")
    # モジュールごとに係数や rank を変える指定は、この実装が一律 alpha/r で焼くので取りこぼす。
    # 「知らない指定を黙って無視する」のが一番危ないので、空でなければ落とす。
    for key in ("alpha_pattern", "rank_pattern"):
        if acfg.get(key):
            raise SystemExit(f"{key} が空でない: {acfg[key]}\n"
                             "モジュールごとに係数/rank を変える指定はこの実装が扱えない。"
                             "一律 alpha/r で焼くと違う重みになる。")
    if acfg.get("exclude_modules"):
        raise SystemExit(f"exclude_modules が空でない: {acfg['exclude_modules']}\n"
                         "除外指定はこの実装が見ていない。adapter 側のテンソルだけを見て焼くため、"
                         "除外の意図と食い違う可能性がある。")
    if acfg.get("modules_to_save"):
        raise SystemExit(f"modules_to_save が空でない: {acfg['modules_to_save']}\n"
                         "LoRA 以外に丸ごと差し替えるモジュールがある構成は未対応。")
    if acfg.get("use_rslora"):
        # rsLoRA は係数が alpha/sqrt(r)
        coef = lora_alpha / (r ** 0.5)
        scaling_note = f"rsLoRA: alpha/sqrt(r) = {coef:.6f}"
    else:
        coef = lora_alpha / r
        scaling_note = f"LoRA: alpha/r = {coef:.6f}"
    coef *= scale

    ad_p = adapter_dir / "adapter_model.safetensors"
    if not ad_p.is_file():
        raise SystemExit(f"adapter_model.safetensors が無い: {ad_p}")

    # --- adapter を読む（小さいので全部載せてよい）
    pairs: dict[str, dict[str, "torch.Tensor"]] = {}
    with safe_open(str(ad_p), framework="pt") as f:
        ad_keys = list(f.keys())
        for k in ad_keys:
            got = adapter_key_to_module(k)
            if got is None:
                raise SystemExit(f"adapter に LoRA でないテンソルがある: {k}\n"
                                 "（modules_to_save がある構成は未対応。手で確認すること）")
            path, ab = got
            pairs.setdefault(path, {})[ab] = f.get_tensor(k)
    bad = [p for p, d in pairs.items() if set(d) != {"A", "B"}]
    if bad:
        raise SystemExit(f"A と B が揃っていないモジュールがある: {bad[:5]}")

    base_w = base_dir / "model.safetensors"
    if not base_w.is_file():
        raise SystemExit(f"base の model.safetensors が無い: {base_w}")

    # 出力先が base と同じ（または base の中）なら、base を壊したうえで
    # 自己検査が「自分と自分」を比べることになり、壊したことに気付けない。
    try:
        b_res, o_res = base_dir.resolve(), out_dir.resolve()
    except OSError:
        b_res, o_res = base_dir, out_dir
    if b_res == o_res or b_res in o_res.parents or o_res in b_res.parents:
        raise SystemExit(f"--out が --base と重なっている（base={b_res} out={o_res}）。\n"
                         "base を上書きすると元の重みが失われ、検証も意味を失う。別の場所を指定すること。")

    # --- base のキー一覧を見て対応付けを先に確定させる（1 つでも欠けたらここで落とす）
    with safe_open(str(base_w), framework="pt") as f:
        base_keys = set(f.keys())
    mapping: dict[str, str] = {}
    missing: list[str] = []
    for path in pairs:
        hit = next((c for c in candidate_base_keys(path) if c in base_keys), None)
        if hit is None:
            missing.append(path)
        else:
            mapping[path] = hit
    if missing:
        sample = "\n  ".join(missing[:10])
        raise SystemExit(
            f"base に対応する重みが見つからない LoRA モジュールが {len(missing)} 個ある:\n  {sample}\n"
            "黙って飛ばすと『焼いたつもりで素の重み』になるので中止する。\n"
            "base ckpt の階層名を確認すること（例: model.paligemma_with_expert.gemma_expert....）")

    report = {
        "adapter": str(adapter_dir), "base": str(base_dir), "out": str(out_dir),
        "r": r, "lora_alpha": lora_alpha, "scale": scale, "coef": coef,
        "scaling": scaling_note, "n_modules": len(pairs),
        "n_adapter_tensors": len(ad_keys), "n_base_tensors": len(base_keys),
        "modules": sorted(mapping.items())[:8],
    }
    if dry_run:
        report["dry_run"] = True
        return report

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- base を 1 テンソルずつ読みながら、対象だけ足して新しい dict を作る
    inv = {v: k for k, v in mapping.items()}
    touched, deltas = 0, []
    tensors: dict[str, "torch.Tensor"] = {}
    with safe_open(str(base_w), framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            t = f.get_tensor(k)
            if k in inv:
                path = inv[k]
                A = pairs[path]["A"].to(torch.float32)   # (r, in)
                B = pairs[path]["B"].to(torch.float32)   # (out, r)
                if A.shape[0] != r or B.shape[1] != r:
                    raise SystemExit(f"{path}: LoRA の形が r={r} と合わない A={tuple(A.shape)} B={tuple(B.shape)}")
                dw = (B @ A) * coef
                if dw.shape != t.shape:
                    raise SystemExit(f"{path}: 差分の形が本体と合わない dw={tuple(dw.shape)} W={tuple(t.shape)}")
                orig_dtype = t.dtype
                w = t.to(torch.float32) + dw
                rel = float(dw.norm() / (t.to(torch.float32).norm() + 1e-12))
                deltas.append((path, rel))
                tensors[k] = w.to(orig_dtype)
                touched += 1
            else:
                tensors[k] = t

    if touched != len(mapping):
        raise SystemExit(f"焼き込んだテンソル数が合わない: {touched} != {len(mapping)}")

    save_file(tensors, str(out_dir / "model.safetensors"), metadata=meta or None)
    del tensors

    # --- 同梱物は base から持ってくる（**正規化統計は配布のものを使う**）
    copied = []
    for name in sorted(os.listdir(base_dir)):
        if name == "model.safetensors":
            continue
        src, dst = base_dir / name, out_dir / name
        if dst.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=False)
        elif link_mode == "hardlink":
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        copied.append(name)

    # --- 読み直し検証: 対象は変わっていること、非対象はビット一致であること
    ok_changed, ok_same = 0, 0
    with safe_open(str(base_w), framework="pt") as fb, \
         safe_open(str(out_dir / "model.safetensors"), framework="pt") as fo:
        okeys = set(fo.keys())
        if okeys != base_keys:
            raise SystemExit("出力のテンソル集合が base と違う")
        for k in list(mapping.values()):
            if torch.equal(fb.get_tensor(k), fo.get_tensor(k)):
                raise SystemExit(f"焼き込んだはずの {k} が base とビット一致している（scale が 0 か、足せていない）")
            ok_changed += 1
        for k in list(base_keys - set(mapping.values()))[:64]:
            if not torch.equal(fb.get_tensor(k), fo.get_tensor(k)):
                raise SystemExit(f"触っていないはずの {k} が変わっている")
            ok_same += 1

    deltas.sort(key=lambda x: -x[1])
    report.update({
        "touched": touched, "copied": copied,
        "verified_changed": ok_changed, "verified_identical_sample": ok_same,
        "rel_delta_max": deltas[0] if deltas else None,
        "rel_delta_min": deltas[-1] if deltas else None,
        "rel_delta_mean": (sum(d for _, d in deltas) / len(deltas)) if deltas else None,
    })
    return report


# ---------------------------------------------------------------- 自己検査

def selftest() -> int:
    ng = 0

    def eq(got, want, name):
        nonlocal ng
        if got != want:
            print(f"  NG {name}: got={got!r} want={want!r}")
            ng += 1
        else:
            print(f"  ok {name}")

    print("[selftest] キーの対応付け")
    eq(adapter_key_to_module("base_model.model.model.state_proj.lora_A.weight"),
       ("model.state_proj", "A"), "PEFT 標準形")
    eq(adapter_key_to_module("base_model.model.a.b.self_attn.q_proj.lora_B.default.weight"),
       ("a.b.self_attn.q_proj", "B"), ".default 入り")
    eq(adapter_key_to_module("something.weight"), None, "LoRA でないキー")
    eq(candidate_base_keys("model.state_proj")[0], "model.state_proj.weight", "候補の先頭")
    eq("state_proj.weight" in candidate_base_keys("model.state_proj"), True, "前置き剥がしも候補に入る")

    print("[selftest] 係数")
    eq(round(8 / 16, 6), 0.5, "alpha/r")

    print(f"[selftest] {'すべて通った' if ng == 0 else f'{ng} 件 NG'}")
    return 0 if ng == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="公開 ckpt に LoRA を焼き込んで完全チェックポイントを作る")
    ap.add_argument("--base", help="公開 ckpt の model_weights ディレクトリ")
    ap.add_argument("--adapter", help="LoRA アダプタのディレクトリ")
    ap.add_argument("--out", help="出力する model_weights ディレクトリ")
    ap.add_argument("--scale", type=float, default=1.0, help="寄与の強さ。1.0 が PEFT の merge と同一")
    ap.add_argument("--dry-run", action="store_true", help="対応付けの検査だけ行い書かない")
    # 既定は copy。ディスクを節約したい測定用ビルドのときだけ明示的に hardlink を選ぶこと。
    ap.add_argument("--link-mode", choices=("hardlink", "copy"), default="copy")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    for need in ("base", "adapter", "out"):
        if not getattr(a, need):
            ap.error(f"--{need} が要る")
    if not (0.0 < a.scale <= 4.0):
        ap.error("--scale は 0 より大きく 4 以下")

    rep = merge(Path(a.base).expanduser(), Path(a.adapter).expanduser(),
                Path(a.out).expanduser(), a.scale, a.dry_run, a.link_mode)
    print(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
