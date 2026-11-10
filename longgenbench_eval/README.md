# longgenbench_eval

Judge and score LongGenBench generations (Wu et al., ICLR 2025, arXiv:2409.02076)
the way the paper did. The harness is self-contained: it imports nothing from the
repository around it, so you can copy this directory next to any baseline's code.

* `data.py`: the pinned dataset (`mozhu/LongGenBench` @ `3c96cac`, SHA-256 checked).
* `judge.py`: the paper's judge (Llama-3.1-8B-Instruct, vLLM, batched).
* `score.py`: CR, STIC-1, STIC-2, plus Table 2's Len and wAvg.
* `metrics.py`, `reference.py`: the definitions, and every constant with its source.

`metrics.py`, `data.py` and `score.py` need only the Python standard library.
`judge.py` needs `vllm` (`requirements-judge.txt`).

## What "as the paper did" means

The paper's Table 2 is the authors' `static.csv`. The code that produced it,
read from the repository history (`mozhu621/LongGenBench`), is:

| step | reference code | reproduced here |
|---|---|---|
| generation | `inference.py` @ `7b59ec6`: vLLM on the raw prompt, `temperature=0.95, top_p=0.95, max_tokens=16000, seed=6211027, stop='*** finished'`, output `prefix + text` | your generation code (see below) |
| judge | `eval.py` @ `951769c`: Llama-3.1-8B-Instruct, zero-shot `"Context: " + block + "### Instruction: Does this context include the {X}? Please answer with 'yes' or 'no' only."`, `temperature=0.95, top_p=0.95, max_tokens=50, seed=42`, verdict `'yes' in response.strip().lower()` | `judge.py`, character for character |
| metrics | `eval.py` (block parsing, CR), `static.ipynb` (STIC-1, STIC-2, Len) | `metrics.py` |

Two places where the paper's text disagrees with its code, and the code wins:

* The paper says "greedy decoding". Every `inference.py` that produced results sampled at T = 0.95.
* Today's `eval.py` has a few-shot prompt and Llama-3.3-70B. Both arrived on 2025-02-06, after the paper.
  They are available as `--prompt 2025 --model ...`, but they are not the default.

Checked: the formulas reproduce 16 of the 18 rows of the paper's Table 2 from the
authors' own counts, digit for digit. The two exceptions are the GPT-4o rows, which
carry typos in the paper itself.

## Generating for a baseline

The judge and scorer only see your outputs, so a fair comparison generates the way
the paper did:

* the dataset `prompt` verbatim, as raw text (one BOS, **no** chat template). It ends
  with the first entry's header, e.g. `#*# Week 1 (January 1st - January 7th):`;
* stop at `*** finished` (excluded), at EOS, or after 16,000 new tokens (32,000 at 32K);
* temperature 0.95, top_p 0.95, no top-k;
* write `output = prefix + generated_text`.

## Input format

A JSONL file, one line per generated document:

```json
{"id": 0, "output": "#*# Week 1 (January 1st - January 7th):\n..."}
```

* `id` is the document's index in `Dataset_short.json` (16K) or `Dataset_long.json` (32K).
  `data.first_n_ids(...)` gives the first N of each type.
* `output` is `prefix + generation`; use `null` for a failed generation, which scores 0.
* Other fields are ignored.

The reference's own `inference.py` output (a JSON list with `input` and
`output_blocks`) is accepted as is.

## Commands

```bash
python -m longgenbench_eval.data --out data/longgenbench
```

Judging runs in an environment with `vllm` (`requirements-judge.txt`: 0.8.5.post1, on its
V0 engine by default; the paper ran 0.5.4). The judge can live in its own virtualenv; it
does not need your generation stack:

```bash
uv venv --python 3.12 judge_env
```

```bash
uv pip install --python judge_env/bin/python -r longgenbench_eval/requirements-judge.txt
```

Then judge with that environment's Python:

```bash
judge_env/bin/python -m longgenbench_eval.judge --setting 16K --data data/longgenbench --generations runs/ours.jsonl runs/baseline.jsonl
```

Scoring needs only the standard library:

```bash
python -m longgenbench_eval.score --setting 16K --data data/longgenbench --run ours runs/ours.jsonl --run baseline runs/baseline.jsonl --json scores.json
```

`judge` writes `X.verdicts.jsonl` (every judge response) and `X.verdicts.meta.json`
next to each `X.jsonl`. It skips files whose verdicts are already up to date. `score`
refuses verdicts made from a different version of the generations.
