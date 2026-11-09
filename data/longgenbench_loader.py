"""LongGenBench data, pinned: the benchmark exactly as SCOPE packaged and ran it.

**What LongGenBench is.** Liu, Dong, Hu & Chu, *LongGenBench: Long-context
Generation Benchmark* (EMNLP 2024 Findings, arXiv:2410.04199). K questions from
GSM8K, MMLU or CommonsenseQA are concatenated into ONE prompt after a few worked
examples, and the model must answer all of them, in order, in ONE long
generation (``Answer_9: ... The answer is 18.`` and so on). The prompt is a few
thousand tokens; the answer is the long part. That makes it a test of the decode
phase, which is where this cache evicts: every token after the first is produced
from the budgeted cache (``first_eviction_step = 0``).

**Why SCOPE's packaging.** The original release (``Dominic789654/LongGenBench``)
only drives the OpenAI API. The KV-compression version, with Llama-3.1-8B-Instruct
under HF ``generate`` and published full-cache numbers, is SCOPE (Wu et al.,
*SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation*, ACL
2025, ``Linking-ai/SCOPE``). Its ``run_longgenbench.py`` /
``eval_longgenbench.py`` define the protocol this module and
``modules/evaluation/longgenbench_runner.py`` reproduce, and its
``results_longgenbench_{4K,8K}/ALLKV+None...`` files are the full-cache reference
a reproduction is checked against. Everything is fetched from one pinned commit
and checked against a SHA-256, so two runs can only differ in the method.

Two settings, as SCOPE names them:

=======  ======  ======  ======  ===============
setting  GSM8K   MMLU    CSQA    max_new_tokens
=======  ======  ======  ======  ===============
``4K``   K = 30  K = 30  K = 40  4,096
``8K``   K = 60  K = 60  K = 80  7,950
=======  ======  ======  ======  ===============

Two quirks of the reference are reproduced on purpose, because the published
numbers include them:

* The MMLU system prompt ends in ``"... questions (with answers) about "`` with
  no subject. ``run_longgenbench.py`` appends ``example["task"]`` to the template
  only AFTER it has formatted the system prompt from it, so the subject never
  reaches the model.
* ``run_longgenbench.py`` drops MMLU rows 21, 30, 31 and 48 in both settings
  ("to avoid OOM" on its 80 GB A100). ``drop_scope_oom=True`` (default) keeps
  the comparison to its table; ``False`` scores all 57 subjects.

Usage::

    python -m data.longgenbench_loader --out data/longgenbench --setting 8K

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
from typing import Dict, Iterable, List, Optional, Sequence

SCOPE_REPO = "Linking-ai/SCOPE"
#: The commit every file below was read and hashed at.
SCOPE_COMMIT = "8cbae48042e34da630c7a044f535de7a991c2999"
RAW_BASE = f"https://raw.githubusercontent.com/{SCOPE_REPO}/{SCOPE_COMMIT}/"

TASKS = ("gsm8k", "mmlu", "csqa")
SETTINGS = ("4K", "8K")

#: Questions per prompt, per setting (``run_longgenbench.py --K``; CSQA is K*4/3).
K_PER_TASK = {
    "4K": {"gsm8k": 30, "mmlu": 30, "csqa": 40},
    "8K": {"gsm8k": 60, "mmlu": 60, "csqa": 80},
}

#: ``dataset2maxlen_4K`` / ``dataset2maxlen_8K`` in ``run_longgenbench.py``.
MAX_NEW_TOKENS = {"4K": 4096, "8K": 7950}

#: The worked examples take ``Question_1 .. Question_{n-1}``, so the first real
#: question is ``Question_n`` and its answer ``Answer_n`` (8 GSM8K demos, 5 for
#: MMLU and CSQA). ``eval_longgenbench.py`` labels them the same way.
FIRST_QUESTION_INDEX = {"gsm8k": 9, "mmlu": 6, "csqa": 6}

#: MMLU rows ``run_longgenbench.py`` removes before generating, in both settings.
SCOPE_DROPPED_MMLU = (21, 30, 31, 48)

#: ``model2prompt`` in ``run_longgenbench.py``, verbatim. The MMLU entry ends in
#: "about " with nothing after it; see the module docstring.
SYSTEM_PROMPTS = {
    "gsm8k": ("Answer each question step by step, adhering to the format shown in the "
              "examples provided. Start each response with 'Answer_' and introduce the "
              "final response with 'The answer is'. Do not repeat the question. Ensure "
              "that you respond to all the questions presented, regardless of their number."),
    "mmlu": ("Answer each question step by step, adhering to the format shown in the "
             "examples provided. Start each response with 'Answer_' and introduce the "
             "final response with 'The answer is'. Do not repeat the question. Ensure "
             "that you respond to all the questions presented, regardless of their number. "
             "The following are multiple choice questions (with answers) about "),
    "csqa": ("Answer each question step by step, adhering to the format shown in the "
             "examples provided. Start each response with 'Answer_' and introduce the "
             "final response with 'The answer is'. Do not repeat the question. Ensure "
             "that you respond to all the questions presented, regardless of their number."),
}

#: The only prompt format implemented: ``build_chat_llama3_modify``, which
#: ``run_longgenbench.py`` uses for every model path containing "llama-3"
#: (Llama-3.1-8B-Instruct included). It is NOT the Llama-3 chat template: there
#: are no header tokens, and the tokenizer adds the one BOS.
PROMPT_FORMAT = "scope_llama3_modify"

#: Data files: relative path in the SCOPE repo -> SHA-256 at ``SCOPE_COMMIT``.
DATA_FILES = {
    "data/LongGenBench/gsm8k/gsm8k_30.jsonl":
        "bc056d62137f87dec862de9e512d3644220a6e1a043edbaf8aa264274a5391bd",
    "data/LongGenBench/gsm8k/gsm8k_60.jsonl":
        "227b4bc4b3bc0f3ec446405178911ef59ca1903e6e62a20d1b6452b704cc32ff",
    "data/LongGenBench/mmlu/mmlu_30.jsonl":
        "d366a549e59f6266618e4590032e7eca1d8a0c0e0418deec4f6585e60f21127d",
    "data/LongGenBench/mmlu/mmlu_60.jsonl":
        "e04703b9e873dc1ad59f10a12129b67e3e55c49dbae0f026e1510ee54c6b43b4",
    "data/LongGenBench/csqa/csqa_40.jsonl":
        "1c5eaa3b202b4dc1a6858f5ef02f5c5e1779ba556210b4d882a13e5983b6096d",
    "data/LongGenBench/csqa/csqa_80.jsonl":
        "7590db0348b4b64d7aa3b6630b22332f63a6e5987e30aca9f6887c8dfd0c58e9",
}

_REF_DIR = {
    "4K": "results_longgenbench_4K/ALLKV+None_decoding_window_512_recent_window_256/"
          "meta-llama-3.1-8b-instruct_2048",
    "8K": "results_longgenbench_8K/ALLKV+None_decoding_window_512_recent_window_256/"
          "meta-llama-3.1-8b-instruct_4096",
}

#: SCOPE's full-cache (``ALLKV``) Llama-3.1-8B-Instruct predictions, one file per
#: (setting, task), with the SHA-256 and the score its ``results.csv`` publishes.
#: The predictions carry the exact prompt SCOPE fed the model, which is how the
#: prompt construction here is checked byte for byte, and they are what a
#: full-cache reproduction is compared against.
REFERENCE_FILES = {
    ("4K", "gsm8k"): (f"{_REF_DIR['4K']}/gsm8k/ALLKV.json",
                      "04358896c62eb5f90dc1ff065f34df8fad031ef7b44be33b1fdf1eff3e3c6a9c", 53.2558),
    ("4K", "mmlu"): (f"{_REF_DIR['4K']}/mmlu/ALLKV.json",
                     "91746069eb57cffd94f5a6738f4aee385a14d2c600037fd2f733cf9916f5a153", 54.4025),
    ("4K", "csqa"): (f"{_REF_DIR['4K']}/csqa/ALLKV.json",
                     "dede505783db06eacd0fa9f33af5086924a729b87dbff91be55496d7e2336979", 71.6667),
    ("8K", "gsm8k"): (f"{_REF_DIR['8K']}/gsm8k/ALLKV.json",
                      "5c1e0fd666a32e9bb7730259eda368ebfdb29851e193e778513ac75ad9449e23", 44.4444),
    ("8K", "mmlu"): (f"{_REF_DIR['8K']}/mmlu/ALLKV.json",
                     "3f3a0e1ff86a2e3fd9244e478ab5bafc3ff2c96244fa32d79424cbf5b44b73e6", 51.0063),
    ("8K", "csqa"): (f"{_REF_DIR['8K']}/csqa/ALLKV.json",
                     "f87883d454d9cfeaddea1c17de7fc3d1b63d5cea00f88f42e8f81f7bedca430c", 64.9167),
}

MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def _check(setting: str, task: str) -> None:
    if setting not in SETTINGS:
        raise ValueError(f"LongGenBench setting must be one of {SETTINGS}, got {setting!r}")
    if task not in TASKS:
        raise ValueError(f"LongGenBench task must be one of {TASKS}, got {task!r}")


def data_relpath(setting: str, task: str) -> str:
    """The data file's path inside the SCOPE repo (and under ``data_dir``)."""
    _check(setting, task)
    return f"data/LongGenBench/{task}/{task}_{K_PER_TASK[setting][task]}.jsonl"


def reference_relpath(setting: str, task: str) -> str:
    _check(setting, task)
    return REFERENCE_FILES[(setting, task)][0]


def published_reference_score(setting: str, task: str) -> float:
    """SCOPE's published full-cache score (``results.csv``) for this cell."""
    _check(setting, task)
    return REFERENCE_FILES[(setting, task)][2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_sha(relpath: str) -> str:
    if relpath in DATA_FILES:
        return DATA_FILES[relpath]
    for path, sha, _ in REFERENCE_FILES.values():
        if path == relpath:
            return sha
    raise KeyError(relpath)


def verified_path(data_dir: str | Path, relpath: str) -> Path:
    """``data_dir / relpath``, after checking it is byte-identical to the pin."""
    path = Path(data_dir) / relpath
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Fetch the pinned files first:\n"
            f"    python -m data.longgenbench_loader --out {data_dir}")
    got, want = _sha256(path), _expected_sha(relpath)
    if got != want:
        raise ValueError(
            f"{path} has SHA-256 {got[:16]}..., not the pinned {want[:16]}... "
            f"(SCOPE @ {SCOPE_COMMIT[:10]}). It is not the benchmark the published "
            "numbers describe; delete it and fetch again.")
    return path


def _read_jsonl(path: Path) -> List[Dict]:
    """One JSON object per ``\\n``-terminated line.

    Not ``str.splitlines()``: that also splits on U+2028, U+0085 and the other
    Unicode line boundaries, which these files carry INSIDE their strings (MMLU
    and CSQA text), so it cuts records in half.
    """
    return [json.loads(line) for line in path.read_bytes().decode("utf-8").split("\n")
            if line.strip()]


def dataset_sha(data_dir: str | Path, setting: str, task: str) -> str:
    return _sha256(verified_path(data_dir, data_relpath(setting, task)))


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def _fetch(relpath: str, dst: Path, retries: int = 4) -> None:
    url = RAW_BASE + urllib.parse.quote(relpath)
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


def fetch(out_dir: str | Path, settings: Iterable[str] = SETTINGS,
          tasks: Iterable[str] = TASKS, with_reference: bool = True) -> Path:
    """Download the pinned data (and SCOPE's full-cache predictions) to *out_dir*.

    Files already present with the right hash are kept. A file with the wrong
    hash is re-downloaded, and if it still disagrees this raises: a mirror that
    serves different bytes is not the benchmark.
    """
    out = Path(out_dir)
    rels = []
    for s in settings:
        for t in tasks:
            rels.append(data_relpath(s, t))
            if with_reference:
                rels.append(reference_relpath(s, t))
    for rel in rels:
        dst = out / rel
        if dst.is_file() and _sha256(dst) == _expected_sha(rel):
            continue
        _fetch(rel, dst)
        verified_path(out, rel)
        print(f"fetched {rel}")
    manifest = {
        "source": f"https://github.com/{SCOPE_REPO}",
        "commit": SCOPE_COMMIT,
        "files": {rel: _expected_sha(rel) for rel in sorted(set(rels))},
    }
    old = {}
    mpath = out / MANIFEST_NAME
    if mpath.is_file():
        old = json.loads(mpath.read_text(encoding="utf-8")).get("files", {})
    manifest["files"] = dict(sorted({**old, **manifest["files"]}.items()))
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# prompts and examples
# ---------------------------------------------------------------------------


def build_prompt(task: str, raw_prompt: str) -> str:
    """The exact string ``run_longgenbench.py`` tokenizes for a Llama-3 model.

    ``build_chat_llama3_modify(system_prompt, prompt)``, with the system prompt
    from ``model2prompt``. The tokenizer then adds the one BOS
    (``add_special_tokens=True``), which ``utils.prompting.encode_prompt`` does
    too, because this string does not start with one.
    """
    if task not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown LongGenBench task {task!r}")
    return f"<<SYS>>\n {SYSTEM_PROMPTS[task]} \n<</SYS>>\n\n{raw_prompt}"


def assert_supported_model(model_name: str) -> None:
    """Only the Llama-3 branch of SCOPE's prompt dispatch is implemented.

    ``run_longgenbench.py`` picks the format from the model path: Llama-2 gets
    ``[INST] <<SYS>>``, a path containing "llama-3" gets
    ``build_chat_llama3_modify``, anything else gets the bare system prompt. A
    different model under this format would be scored on a prompt the reference
    never used, so it is refused rather than run.
    """
    if "llama-3" not in str(model_name).lower():
        raise ValueError(
            f"LongGenBench prompt format {PROMPT_FORMAT!r} is SCOPE's Llama-3 branch; "
            f"model {model_name!r} would get a different prompt in the reference "
            "(it dispatches on the model path). Use a Llama-3 family model, or "
            "implement that branch first.")


def load_longgenbench(
    data_dir: str | Path,
    setting: str,
    task: str,
    drop_scope_oom: bool = True,
    start: int = 0,
    end: Optional[int] = None,
) -> List[Dict]:
    """The examples of one task, in file order, prompts already built.

    Each record::

        {"id": int,            # line index in SCOPE's file (stable across drops)
         "task": str, "subject": str | None,
         "prompt": str,        # build_prompt(...) -- what is tokenized
         "answers": list[str], # gold, as SCOPE stores them
         "n_questions": int}

    ``start`` / ``end`` slice the list AFTER the MMLU drop, so a chunk is a
    contiguous run of the examples SCOPE scores.
    """
    path = verified_path(data_dir, data_relpath(setting, task))
    rows = _read_jsonl(path)
    out = []
    for i, row in enumerate(rows):
        if task == "mmlu" and drop_scope_oom and i in SCOPE_DROPPED_MMLU:
            continue
        answers = list(row["answers"])
        out.append({
            "id": i,
            "task": task,
            "subject": row.get("task"),
            "prompt": build_prompt(task, row["prompt"]),
            "answers": answers,
            "n_questions": len(answers),
        })
    return out[start:end]


def example_ids(data_dir: str | Path, setting: str, task: str,
                drop_scope_oom: bool = True) -> List[int]:
    """The ``id`` of every example a full run of this task scores."""
    return [e["id"] for e in load_longgenbench(data_dir, setting, task, drop_scope_oom)]


def load_reference(data_dir: str | Path, setting: str, task: str) -> List[Dict]:
    """SCOPE's full-cache predictions for this cell, in its file order."""
    return _read_jsonl(verified_path(data_dir, reference_relpath(setting, task)))


def reference_by_id(data_dir: str | Path, setting: str, task: str,
                    ids: Sequence[int]) -> Dict[int, Dict]:
    """SCOPE's prediction for each example id, matched by prompt, not position.

    SCOPE's output file has one row per example it ran, after its MMLU drop.
    Matching on the raw prompt it recorded (rather than on row order) makes the
    pairing independent of that drop; a prompt that matches no example raises.
    """
    data_path = verified_path(data_dir, data_relpath(setting, task))
    raw = [row["prompt"] for row in _read_jsonl(data_path)]
    by_prompt = {build_prompt(task, p): i for i, p in enumerate(raw)}
    out: Dict[int, Dict] = {}
    for row in load_reference(data_dir, setting, task):
        i = by_prompt.get(row["prompt"])
        if i is None:
            raise ValueError(
                f"a SCOPE {setting}/{task} reference prompt matches no example built "
                "here: the prompt construction has drifted from the reference")
        out[i] = row
    return {i: out[i] for i in ids if i in out}


def _cli_main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="data/longgenbench")
    p.add_argument("--setting", choices=SETTINGS + ("all",), default="all")
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--no-reference", action="store_true",
                   help="skip SCOPE's full-cache predictions (the reproduction check needs them)")
    a = p.parse_args()
    settings = SETTINGS if a.setting == "all" else (a.setting,)
    out = fetch(a.out, settings, a.tasks, with_reference=not a.no_reference)
    for s in settings:
        for t in a.tasks:
            n = len(load_longgenbench(out, s, t))
            print(f"{s} {t:6s} {n:3d} examples x {K_PER_TASK[s][t]} questions, "
                  f"max_new_tokens {MAX_NEW_TOKENS[s]}, sha {dataset_sha(out, s, t)[:12]}")


if __name__ == "__main__":
    _cli_main()
