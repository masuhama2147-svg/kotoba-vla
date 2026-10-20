"""サーバとクライアントで共有する観測・行動の直列化（msgpack + 生バイト列）。"""

import msgpack
import numpy as np


def encode_arrays(arrays: dict[str, np.ndarray]) -> bytes:
    return msgpack.packb(
        {k: {"dtype": str(v.dtype), "shape": list(v.shape), "data": np.ascontiguousarray(v).tobytes()}
         for k, v in arrays.items()},
        use_bin_type=True,
    )


def decode_arrays(payload: bytes) -> dict[str, np.ndarray]:
    raw = msgpack.unpackb(payload, raw=False)
    return {k: np.frombuffer(v["data"], dtype=np.dtype(v["dtype"])).reshape(v["shape"]).copy() for k, v in raw.items()}
