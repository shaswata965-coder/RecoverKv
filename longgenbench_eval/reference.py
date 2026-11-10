"""Everything the paper's numbers depend on, pinned to the code that produced them.

Sources: ``mozhu621/LongGenBench``. The paper's Table 2 is the authors'
``Evalution/results/static.csv`` / ``accuracy_results.csv``; the Llama-3.1-8B-
Instruct 16K row first appears there on 2024-08-18 (commit ``9c3e71d``). The code
in force then, and quoted below, is:

* generation: ``Evalution/inference.py`` @ ``7b59ec6`` (2024-08-17, unchanged
  since): vLLM ``llm.generate`` on the raw prompt strings, ``SamplingParams(
  temperature=0.95, top_p=0.95, max_tokens=16000|32000, seed=6211027,
  stop='*** finished')``, output ``prefix + text`` split on ``#*#``;
* judge: ``Evalution/eval.py`` @ ``951769c`` (2024-08-10):
  Meta-Llama-3.1-8B-Instruct under vLLM, a ZERO-shot prompt, ``SamplingParams(
  temperature=0.95, top_p=0.95, max_tokens=50, seed=42)``, verdict ``'yes' if
  'yes' in response.strip().lower()``;
* aggregates: ``Evalution/results/static.ipynb`` (STIC-1, STIC-2, Len) and the
  paper (wAvg = CR x STIC-2).

Two things the paper's text says differently, and why the code wins here:

* The paper says "greedy decoding"; every ``inference.py`` that produced results
  samples at T = 0.95. ``reference`` decoding reproduces the code; ``greedy`` is
  available.
* The paper's Appendix C prompt is the zero-shot one below. The few-shot prompt
  in today's ``eval.py`` arrived on 2025-02-06 with a switch to
  Llama-3.3-70B-Instruct, after the paper; it is kept as ``JUDGE_PROMPT_2025``
  and never used by default.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

HF_DATASET = "mozhu/LongGenBench"
#: The HF dataset revision hashed below (2025-01-23). Dataset_short.json is
#: byte-identical to the one file version ever committed to the GitHub repo.
HF_REVISION = "3c96cac7cd6bc39b4d9893276cbd029f638c257f"
CODE_REPO = "mozhu621/LongGenBench"
GENERATION_CODE = "Evalution/inference.py @ 7b59ec6"
JUDGE_CODE = "Evalution/eval.py @ 951769c"

SETTINGS = ("16K", "32K")
FILES = {
    "16K": ("Dataset_short.json",
            "3cf98c4badc330a93d1010f57f6658b09f2ebc2e6428a3df083b652d902d96e8"),
    "32K": ("Dataset_long.json",
            "f61048da5ecfb1b717584af8b2dc5999198459e9b95aceedde2dcc6f5cdd35e7"),
}
#: Task types in file order -> entries one document must contain.
TYPES = {
    "16K": {"Week": 52, "Floor": 100, "Menu Week": 52, "Block": 100},
    "32K": {"Day": 365, "Floor": 300, "Menu Day": 365, "Block": 361},
}
#: Instructions in the whole file: static.csv "Total Checks" (6205 / 8215).
TOTAL_CHECKS = {"16K": 6205, "32K": 8215}

# ---------------------------------------------------------------------------
# generation (inference.py)
# ---------------------------------------------------------------------------

#: ``MAX_LENGTH`` in Run_short_*.sh / Run_long_*.sh.
MAX_NEW_TOKENS = {"16K": 16000, "32K": 32000}
#: The dataset prompt verbatim, as a raw completion (one BOS, no chat template).
#: It ends with the first entry's header, so the model continues the document.
PROMPT_FORMAT = "raw_completion"
STOP_STRING = "*** finished"        # SamplingParams(stop=...): excluded from the text
BLOCK_SEP = "#*#"                   # process_output: output.split('#*#')
#: SamplingParams of the runs behind Table 2. vLLM's top_k=-1 means no top-k
#: (top_k=0 in transformers); a per-request seed, the same for every document.
GENERATION_SAMPLING = {"temperature": 0.95, "top_p": 0.95, "top_k": 0, "seed": 6211027}

# ---------------------------------------------------------------------------
# judge (eval.py @ 951769c)
# ---------------------------------------------------------------------------

JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"   # the paper's "Llama3.1-8B instruction model"
JUDGE_SAMPLING = {"temperature": 0.95, "top_p": 0.95, "max_tokens": 50, "seed": 42}


def judge_prompt(block: str, event_desc: str) -> str:
    """eval.py @ 951769c, character for character (no space before ``###``)."""
    return ("Context: " + block +
            f"### Instruction: Does this context include the {event_desc}? "
            "Please answer with 'yes' or 'no' only.")


#: eval.py since 2025-02-06 (with Llama-3.3-70B-Instruct). Not the paper's.
JUDGE_EXAMPLES_2025 = [
    "Example 1: Context: The district's new residential area will feature two 10-story apartment buildings, a shopping mall, and a park. The construction is set to begin at 9 AM, and the park will include a playground and jogging tracks. The plan also includes space for a small medical clinic, which will open from 10 AM to 6 PM. ### Instruction: Does this context include a medical clinic? Please answer with 'yes' or 'no' only. Answer: yes",  # noqa: E501
    "Example 2: Context: The menu for today's lunch at the office includes grilled turkey, mashed potatoes with gravy, roasted vegetables, and pumpkin pie for dessert. The meal will be served from 12 PM to 2 PM, and there will be a vegetarian option available. The meal is planned to accommodate 50 people, and the turkey will be served with cranberry sauce. ### Instruction: Does this context include mashed potatoes? Please answer with 'yes' or 'no' only. Answer: yes",  # noqa: E501
    "Example 3: Context: On April 15th, the weather was sunny with a high of 75°F. In the morning, I volunteered for a community cleanup from 9 AM to 12 PM. We collected trash and planted 20 new trees along the riverbank. After lunch, I helped organize the donation of clothes and food for a local shelter, where we served sandwiches and drinks. The day ended at 4 PM. ### Instruction: Does this context include long-distance running? Please answer with 'yes' or 'no' only. Answer: no",  # noqa: E501
]


def judge_prompt_2025(block: str, event_desc: str) -> str:
    """Today's eval.py few-shot prompt. Only with ``--prompt 2025``."""
    return ("\n".join(JUDGE_EXAMPLES_2025) + " \n ### Refer to the examples above for how to answer. \nContext: "
            + block + f"\n\n### Instruction: Now, for the following context, does it include the "
            f"{event_desc}? Please answer with 'yes' or 'no' only. Answer: ")


JUDGE_PROMPTS = {"paper": judge_prompt, "2025": judge_prompt_2025}


def verdict(response: str) -> str:
    """evaluate_accuracy: ``'yes' if 'yes' in response.strip().lower() else 'no'``."""
    return "yes" if "yes" in response.strip().lower() else "no"


# ---------------------------------------------------------------------------
# the paper's Table 2 (CR, STIC-1, STIC-2, Len, wAvg; percent, words)
# ---------------------------------------------------------------------------

#: Re-derived exactly from the authors' static.csv counts in
#: ``tests/test_longgenbench_eval.py`` (16 of 18 rows; the two GPT-4o rows carry
#: typos in the paper itself and are left out).
PAPER_TABLE2 = {
    "16K": {
        "Mistral-7B-Instruct-v0.2": (81.8, 22.0, 17.4, 7296, 14.3),
        "Llama-3.1-8B-Instruct": (93.5, 23.4, 22.0, 8804, 20.6),
        "Qwen2-7B-Instruct": (60.0, 27.9, 16.1, 5138, 9.7),
        "FILM-7B": (36.0, 9.9, 3.9, 6280, 1.4),
        "LongWriter-llama3.1-8b": (46.0, 22.6, 9.8, 11036, 4.5),
        "Mixtral-8x7B-Instruct-v0.1": (83.0, 35.4, 28.0, 8113, 23.3),
        "Llama-3.1-70B-Instruct": (79.3, 24.9, 21.1, 8055, 16.7),
        "Qwen2-72B-Instruct": (94.3, 25.5, 24.0, 8013, 22.7),
    },
    "32K": {
        "Mistral-7B-Instruct-v0.2": (48.2, 37.0, 16.3, 16146, 7.9),
        "Llama-3.1-8B-Instruct": (77.6, 26.5, 18.9, 17354, 14.6),
        "Qwen2-7B-Instruct": (40.0, 31.7, 12.6, 9617, 5.0),
        "FILM-7B": (37.4, 30.9, 10.9, 13775, 4.1),
        "LongWriter-llama3.1-8b": (34.5, 33.6, 10.0, 19925, 3.5),
        "Mixtral-8x7B-Instruct-v0.1": (60.5, 39.9, 22.3, 15839, 13.5),
        "Llama-3.1-70B-Instruct": (63.1, 35.8, 21.7, 15197, 13.7),
        "Qwen2-72B-Instruct": (66.2, 27.5, 17.4, 19845, 11.5),
    },
}
TABLE2_COLUMNS = ("CR", "STIC-1", "STIC-2", "Len", "wAvg")
