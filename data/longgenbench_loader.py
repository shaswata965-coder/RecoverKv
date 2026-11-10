"""LongGenBench (Wu et al., ICLR 2025), pinned: long-form generation from a short prompt.

**What it is.** Wu, Hee, Hu & Lee, *LongGenBench: Benchmarking Long-Form
Generation in Long Context LLMs* (ICLR 2025, arXiv:2409.02076; code
``mozhu621/LongGenBench``, data ``mozhu/LongGenBench`` on the HF Hub). Each
example asks for one long structured document: a weekly diary for a year, a
year of weekly menus, a 100-floor skyscraper, a 100-block city. The prompt is
~600-800 tokens; the answer is up to 16,000 tokens (``16K``, the "short
version") or 32,000 (``32K``, the "long version"). The prompt carries Specific
Task Instructions that must surface in particular entries: once (``Floor 51`` is
a design studio), over a range (weeks 24-25 in Paris) and periodically (a
workshop every 5 weeks from week 13). So it tests whether a model still follows
its instructions thousands of tokens into its own output, which for this
repository is a test of the decode-phase cache: the prompt is tiny, and almost
everything the model attends to is its own generation.

=======  ============================  ==========================  ==============
setting  file                          types (100 examples each)   max_new_tokens
=======  ============================  ==========================  ==============
``16K``  ``Dataset_short.json``        Week (52), Floor (100),     16,000
                                       Menu Week (52), Block (100)
``32K``  ``Dataset_long.json``         Day (365), Floor (300),     32,000
                                       Menu Day (365), Block (361)
=======  ============================  ==========================  ==============

(The number after each type is how many entries one document must contain.)

**The protocol, from the reference code** (``Evalution/inference.py``):

* the dataset ``prompt`` is generated from verbatim, as a raw completion: no chat
  template, one BOS from the tokenizer. The prompt ends with the first entry's
  header (``#*# Week 1 (January 1st - January 7th):``), so the model continues
  the document;
* generation stops at ``*** finished`` (excluded from the text, as vLLM does), at
  the model's EOS, or at ``max_new_tokens``;
* the scored text is ``prefix + generation``, split into entries on ``#*#``.

Decoding: the paper says greedy ("using greedy decoding", §3.1);
``inference.py`` samples (``temperature=0.95, top_p=0.95``). Greedy is the
default here, because it is what the paper reports and because only a
deterministic decode makes ``ours - fullkv`` a statement about the cache rather
than about two draws. ``repo_sampling`` reproduces the code's settings.

This module imports nothing heavier than the standard library, so a notebook can
use it without pulling in torch.
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

HF_DATASET = "mozhu/LongGenBench"
#: The dataset revision every hash below was taken at (HF Hub, 2025-01-23).
HF_REVISION = "3c96cac7cd6bc39b4d9893276cbd029f638c257f"
#: The reference code (prompting, stop string, metric) was read at this commit.
CODE_REPO = "mozhu621/LongGenBench"
CODE_COMMIT = "45c320e11fb5b6589f787011caa4313ff6600275"

SETTINGS = ("16K", "32K")
FILES = {
    "16K": ("Dataset_short.json",
            "3cf98c4badc330a93d1010f57f6658b09f2ebc2e6428a3df083b652d902d96e8"),
    "32K": ("Dataset_long.json",
            "f61048da5ecfb1b717584af8b2dc5999198459e9b95aceedde2dcc6f5cdd35e7"),
}
#: ``MAX_LENGTH`` in ``Run_short_*.sh`` / ``Run_long_*.sh``.
MAX_NEW_TOKENS = {"16K": 16000, "32K": 32000}
#: Task types in file order, and the number of entries each document must have.
TYPES = {
    "16K": {"Week": 52, "Floor": 100, "Menu Week": 52, "Block": 100},
    "32K": {"Day": 365, "Floor": 300, "Menu Day": 365, "Block": 361},
}
#: ``SamplingParams(stop='*** finished')`` in ``inference.py``.
STOP_STRING = "*** finished"
#: ``process_output``: ``output.split('#*#')``.
BLOCK_SEP = "#*#"
#: The dataset prompt verbatim, BOS from the tokenizer, no chat template.
PROMPT_FORMAT = "raw_completion"

#: ``inference.py``'s sampling settings (``repo_sampling``). vLLM's ``top_k=-1``
#: means no top-k; the seed is per run there and per example here.
REPO_SAMPLING = {"temperature": 0.95, "top_p": 0.95, "top_k": 0, "seed": 6211027}
DECODINGS = ("greedy", "repo_sampling")

#: The paper's Table 2 row for Llama-3.1-8B-Instruct (vLLM, bf16), with the
#: per-category accuracies from ``Evalution/results/accuracy_results.csv``.
#: Context for a reader, not a reproduction target: that run used bf16, vLLM,
#: all 400 examples, and its own decoding.
PUBLISHED_LLAMA31_8B = {
    "16K": {"CR": 93.5, "STIC-1": 23.4, "STIC-2": 22.0, "Len": 8804, "wAvg": 20.6,
            "acc_once": 30.47, "acc_range": 33.33, "acc_periodic": 14.65},
    "32K": {"CR": 77.6, "STIC-1": 26.5, "STIC-2": 18.9, "Len": 17354, "wAvg": 14.6,
            "acc_once": 41.67, "acc_range": 28.29, "acc_periodic": 16.42},
}

MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def _check_setting(setting: str) -> None:
    if setting not in SETTINGS:
        raise ValueError(f"LongGenBench setting must be one of {SETTINGS}, got {setting!r}")


def check_type(setting: str, type_: str) -> None:
    _check_setting(setting)
    if type_ not in TYPES[setting]:
        raise ValueError(f"{setting} LongGenBench types are {list(TYPES[setting])}, got {type_!r}")


def type_slug(type_: str) -> str:
    """``"Menu Week"`` -> ``"menu_week"``: file names for a type's outputs."""
    return type_.lower().replace(" ", "_")


def data_filename(setting: str) -> str:
    _check_setting(setting)
    return FILES[setting][0]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verified_path(data_dir: str | Path, setting: str) -> Path:
    """The setting's data file, after checking it is byte-identical to the pin."""
    _check_setting(setting)
    name, want = FILES[setting]
    path = Path(data_dir) / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Fetch the pinned file first:\n"
            f"    python -m data.longgenbench_loader --out {data_dir} --setting {setting}")
    got = _sha256(path)
    if got != want:
        raise ValueError(
            f"{path} has SHA-256 {got[:16]}..., not the pinned {want[:16]}... "
            f"({HF_DATASET} @ {HF_REVISION[:10]}). It is not the benchmark; delete it "
            "and fetch again.")
    return path


def dataset_sha(data_dir: str | Path, setting: str) -> str:
    return _sha256(verified_path(data_dir, setting))


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def _fetch(url: str, dst: Path, retries: int = 4) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dst)
            return
        except Exception as exc:  # network: retry with backoff, then raise
            last = exc
            if attempt < retries:
                time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


def fetch(out_dir: str | Path, settings: Sequence[str] = SETTINGS) -> Path:
    """Download the pinned data file(s) to *out_dir*; keep ones already correct."""
    out = Path(out_dir)
    for s in settings:
        name, want = FILES[s]
        dst = out / name
        if dst.is_file() and _sha256(dst) == want:
            continue
        url = (f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}/"
               f"{urllib.parse.quote(name)}")
        _fetch(url, dst)
        verified_path(out, s)
        print(f"fetched {name}")
    (out / MANIFEST_NAME).write_text(json.dumps({
        "dataset": HF_DATASET, "revision": HF_REVISION,
        "code": CODE_REPO, "code_commit": CODE_COMMIT,
        "files": {FILES[s][0]: FILES[s][1] for s in SETTINGS if (out / FILES[s][0]).is_file()},
    }, indent=2) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# examples
# ---------------------------------------------------------------------------


def load_all(data_dir: str | Path, setting: str) -> List[Dict]:
    """Every example, in file order. ``id`` is the index in the file."""
    rows = json.loads(verified_path(data_dir, setting).read_text(encoding="utf-8"))
    out = []
    for i, r in enumerate(rows):
        out.append({
            "id": i,
            "type": r["type"],
            "number": int(r["number"]),
            "prefix": r["prefix"],
            "prompt": r["prompt"],
            "checks_once": dict(r["checks_once"]),
            "checks_range": dict(r["checks_range"]),
            "checks_periodic": dict(r["checks_periodic"]),
        })
    return out


def load_type(data_dir: str | Path, setting: str, type_: str,
              start: int = 0, end: Optional[int] = None) -> List[Dict]:
    """The examples of one type, in file order, sliced ``[start:end]``.

    The slice is over that type's examples only, so ``start=0, end=5`` is the
    first five Week examples whatever their positions in the file are.
    """
    check_type(setting, type_)
    rows = [e for e in load_all(data_dir, setting) if e["type"] == type_]
    return rows[start:end]


def example_ids(data_dir: str | Path, setting: str, type_: str,
                n: Optional[int] = None) -> List[int]:
    """The ``id`` of the first *n* examples of a type (all when ``n`` is None)."""
    return [e["id"] for e in load_type(data_dir, setting, type_, 0, n)]


def _cli_main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="data/longgenbench")
    p.add_argument("--setting", choices=SETTINGS + ("all",), default="all")
    a = p.parse_args()
    settings = SETTINGS if a.setting == "all" else (a.setting,)
    out = fetch(a.out, settings)
    for s in settings:
        rows = load_all(out, s)
        counts = {t: sum(r["type"] == t for r in rows) for t in TYPES[s]}
        n_checks = sum(len(r["checks_once"]) + len(r["checks_range"]) + len(r["checks_periodic"])
                       for r in rows)
        print(f"{s}: {len(rows)} examples {counts}, {n_checks} checks, "
              f"max_new_tokens {MAX_NEW_TOKENS[s]}, sha {dataset_sha(out, s)[:12]}")


if __name__ == "__main__":
    _cli_main()
