"""The pinned LongGenBench data: fetch, verify, load. Standard library only.

    python -m longgenbench_eval.data --out data/longgenbench            # both settings
    python -m longgenbench_eval.data --out data/longgenbench --setting 16K
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .reference import FILES, HF_DATASET, HF_REVISION, SETTINGS, TOTAL_CHECKS, TYPES


def check_setting(setting: str) -> None:
    if setting not in SETTINGS:
        raise ValueError(f"LongGenBench setting must be one of {SETTINGS}, got {setting!r}")


def check_type(setting: str, type_: str) -> None:
    check_setting(setting)
    if type_ not in TYPES[setting]:
        raise ValueError(f"{setting} LongGenBench types are {list(TYPES[setting])}, got {type_!r}")


def type_slug(type_: str) -> str:
    """``"Menu Week"`` -> ``"menu_week"``."""
    return type_.lower().replace(" ", "_")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verified_path(data_dir: str | Path, setting: str) -> Path:
    """The setting's data file, after checking it is byte-identical to the pin."""
    check_setting(setting)
    name, want = FILES[setting]
    path = Path(data_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Fetch it first:\n"
                                f"    python -m longgenbench_eval.data --out {data_dir}")
    got = sha256_file(path)
    if got != want:
        raise ValueError(f"{path} has SHA-256 {got[:16]}..., not the pinned {want[:16]}... "
                         f"({HF_DATASET} @ {HF_REVISION[:10]}): not the benchmark. Delete it and fetch again.")
    return path


def dataset_sha(data_dir: str | Path, setting: str) -> str:
    return sha256_file(verified_path(data_dir, setting))


def fetch(out_dir: str | Path, settings: Sequence[str] = SETTINGS, retries: int = 4) -> Path:
    """Download the pinned file(s) to *out_dir*; keep ones already correct."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in settings:
        name, want = FILES[s]
        dst = out / name
        if dst.is_file() and sha256_file(dst) == want:
            continue
        url = (f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}/"
               f"{urllib.parse.quote(name)}")
        tmp, last = dst.with_suffix(".part"), None
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                    for chunk in iter(lambda: r.read(1 << 16), b""):
                        f.write(chunk)
                tmp.replace(dst)
                break
            except Exception as exc:  # network: retry with backoff, then raise
                last = exc
                if attempt == retries:
                    raise RuntimeError(f"could not fetch {url}: {last}")
                time.sleep(2 ** (attempt + 1))
        verified_path(out, s)
        print(f"fetched {name}")
    return out


def load_all(data_dir: str | Path, setting: str) -> List[Dict]:
    """Every document, in file order; ``id`` is the index in the file."""
    rows = json.loads(verified_path(data_dir, setting).read_text(encoding="utf-8"))
    return [{"id": i, "type": r["type"], "number": int(r["number"]), "prefix": r["prefix"],
             "prompt": r["prompt"], "checks_once": dict(r["checks_once"]),
             "checks_range": dict(r["checks_range"]), "checks_periodic": dict(r["checks_periodic"])}
            for i, r in enumerate(rows)]


def load_type(data_dir: str | Path, setting: str, type_: str,
              start: int = 0, end: Optional[int] = None) -> List[Dict]:
    """One type's documents, in file order, sliced ``[start:end]`` within that type."""
    check_type(setting, type_)
    return [d for d in load_all(data_dir, setting) if d["type"] == type_][start:end]


def first_n_ids(data_dir: str | Path, setting: str, n: Optional[int] = None,
                types: Optional[Sequence[str]] = None) -> List[int]:
    """The ids of the first *n* documents of each type (all when ``n`` is None)."""
    out: List[int] = []
    for t in (types or TYPES[setting]):
        out += [d["id"] for d in load_type(data_dir, setting, t, 0, n)]
    return out


def _cli_main() -> None:
    p = argparse.ArgumentParser(description="Fetch and verify the pinned LongGenBench data.")
    p.add_argument("--out", default="data/longgenbench")
    p.add_argument("--setting", choices=SETTINGS + ("all",), default="all")
    a = p.parse_args()
    settings = SETTINGS if a.setting == "all" else (a.setting,)
    out = fetch(a.out, settings)
    for s in settings:
        rows = load_all(out, s)
        n = sum(len(r["checks_once"]) + len(r["checks_range"]) + len(r["checks_periodic"]) for r in rows)
        assert n == TOTAL_CHECKS[s], (s, n)
        counts = {t: sum(r["type"] == t for r in rows) for t in TYPES[s]}
        print(f"{s}: {len(rows)} documents {counts}, {n} instructions (= the authors' total), "
              f"sha {dataset_sha(out, s)[:12]}")


if __name__ == "__main__":
    _cli_main()
