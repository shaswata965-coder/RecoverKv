"""Fetch individual tensors from a Hugging Face safetensors checkpoint by HTTP range reads.

Reads each shard's JSON header once, then only the bytes of the tensors asked
for -- so a 7B model's bias vectors cost a few MB, and a layer-by-layer CPU
forward never holds more than one layer. Needs huggingface.co reachable.
"""
import json, os, struct, urllib.request, functools
import numpy as np
import torch

CACHE = os.environ.get("HF_RANGE_CACHE", os.path.expanduser("~/.cache/recoverkv_hf_range"))
os.makedirs(CACHE, exist_ok=True)


def _get(url, start=None, end=None):
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:  # transient proxy resets
            if attempt == 4:
                raise
            import time; time.sleep(2 ** attempt)


class Repo:
    def __init__(self, repo):
        self.base = f"https://huggingface.co/{repo}/resolve/main/"
        tag = repo.replace("/", "__")
        idx_path = os.path.join(CACHE, tag + ".index.json")
        if not os.path.exists(idx_path):
            open(idx_path, "wb").write(_get(self.base + "model.safetensors.index.json"))
        self.index = json.load(open(idx_path))["weight_map"]
        self.tag = tag

    @functools.lru_cache(maxsize=None)
    def header(self, fname):
        path = os.path.join(CACHE, f"{self.tag}__{fname}.header.json")
        if os.path.exists(path):
            h = json.load(open(path))
        else:
            n = struct.unpack("<Q", _get(self.base + fname, 0, 7))[0]
            h = json.loads(_get(self.base + fname, 8, 8 + n - 1))
            h["__data_start__"] = 8 + n
            json.dump(h, open(path, "w"))
        return h

    def tensor(self, name):
        fname = self.index[name]
        h = self.header(fname)
        meta = h[name]
        s, e = meta["data_offsets"]
        raw = _get(self.base + fname, h["__data_start__"] + s, h["__data_start__"] + e - 1)
        assert len(raw) == e - s, (name, len(raw), e - s)
        dt = meta["dtype"]
        if dt == "BF16":
            a = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            t = torch.from_numpy(a.view(np.float32).copy())
        elif dt == "F16":
            t = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).copy()).float()
        elif dt == "F32":
            t = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())
        else:
            raise ValueError(dt)
        return t.reshape(meta["shape"])
