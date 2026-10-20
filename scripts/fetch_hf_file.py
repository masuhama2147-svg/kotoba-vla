"""HuggingFace の大きなファイルを分割並列で取得し、sha256 で検証する。

細い回線で 7.5GB の π0.5 を落とすための道具。分割ごとに途中再開でき、
最後に HF が公開している LFS の sha256 と照合してから本来の名前へ移す。

  python scripts/fetch_hf_file.py lerobot/pi05_libero_finetuned_v044 --out weights/pi05_ft
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def fetch_range(url: str, part: Path, start: int, end: int) -> None:
    want = end - start + 1
    for attempt in range(1000):
        have = part.stat().st_size if part.exists() else 0
        if have >= want:
            return
        # curl の出力をそのまま追記する（途中で切れても、受け取った分は次回の再開位置になる）
        with part.open("ab") as f:
            r = subprocess.run(["curl", "-sSL", "--fail", "--speed-time", "60", "--speed-limit", "1000",
                                "-r", f"{start + have}-{end}", url], stdout=f)
        if r.returncode != 0:
            time.sleep(min(30, 2 + attempt))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--file", default="model.safetensors")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parts", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    info = HfApi().model_info(args.repo, files_metadata=True)
    big = None
    for s in info.siblings:
        if s.rfilename == args.file:
            big = s
        else:  # 小さいファイルは普通に取る
            p = hf_hub_download(args.repo, s.rfilename, local_dir=args.out)
            print("small:", p, flush=True)
    size, sha = big.size, big.lfs.sha256
    url = f"https://huggingface.co/{args.repo}/resolve/{info.sha}/{args.file}"
    step = -(-size // args.parts)
    ranges = [(i * step, min(size, (i + 1) * step) - 1) for i in range(args.parts)]
    parts = [args.out / f".{args.file}.part{i}" for i in range(args.parts)]
    print(f"{args.file}: {size / 1e9:.2f} GB in {args.parts} parts", flush=True)
    with cf.ThreadPoolExecutor(args.parts) as ex:
        futs = [ex.submit(fetch_range, url, p, a, b) for p, (a, b) in zip(parts, ranges)]
        for f in futs:
            f.result()

    target = args.out / args.file
    h = hashlib.sha256()
    with target.with_suffix(".tmp").open("wb") as out:
        for p in parts:
            with p.open("rb") as f:
                while chunk := f.read(1 << 24):
                    h.update(chunk)
                    out.write(chunk)
    if h.hexdigest() != sha:
        raise SystemExit(f"sha256 不一致: {h.hexdigest()} != {sha}")
    target.with_suffix(".tmp").rename(target)
    for p in parts:
        p.unlink()
    print("FETCH_DONE", target, sha, flush=True)


if __name__ == "__main__":
    main()
