"""LongBench v1 data loader — streams the 16 English datasets.

Loads from HuggingFace ``THUDM/LongBench`` with one config name per
dataset.  Supports both the standard and ``_e`` (length-stratified) variants.

Each example has a fixed schema across all 16 datasets::

    {
        "input": str,       # the question or query
        "context": str,     # the long context
        "answers": list,    # list of acceptable reference answers
        "length": int,      # approximate token count
        "dataset": str,     # dataset name
        "language": str,    # "en" or "zh"
        "all_classes": list, # class labels (classification tasks) or null
        "_id": str,         # unique example id
    }
"""

from __future__ import annotations

from typing import List, Optional

from utils.logger import get_logger

log = get_logger(__name__)

# The 16 English datasets used by DefensiveKV Table 2.
LONGBENCH_EN_DATASETS: List[str] = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

# Chinese datasets (excluded by default; DefensiveKV is English-focused).
LONGBENCH_ZH_DATASETS: List[str] = [
    "multifieldqa_zh",
    "dureader",
    "vcsum",
    "lsht",
    "passage_retrieval_zh",
]

# LongBench-E variants (13 English subsets, length-stratified).
LONGBENCH_E_DATASETS: List[str] = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

# Task category groupings (matches DefensiveKV Figure 5).
TASK_CATEGORIES = {
    "Single-doc QA": ["narrativeqa", "qasper", "multifieldqa_en"],
    "Multi-doc QA": ["hotpotqa", "2wikimqa", "musique"],
    "Summarization": ["gov_report", "qmsum", "multi_news"],
    "Few-shot": ["trec", "triviaqa", "samsum"],
    "Synthetic": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}


LONGBENCH_LOCAL_DIR: str = (
    "/home/ee/visitor/man_misn.visitor/kv_cache/LongBenchDataset"
)


def load_longbench_dataset(
    dataset_name: str,
    use_e_variant: bool = False,
    streaming: bool = False,
    data_dir: Optional[str] = None,
):
    """Load a single LongBench dataset, preferring local JSONL files.

    Parameters
    ----------
    dataset_name : str
        One of the 16+5 LongBench dataset config names.
    use_e_variant : bool
        If True, load the ``_e`` length-stratified variant (HuggingFace only).
    streaming : bool
        If True, return an iterable dataset (HuggingFace path only).
    data_dir : str, optional
        Directory containing ``<dataset_name>.jsonl`` files.  Defaults to
        ``LONGBENCH_LOCAL_DIR``.  Pass ``None`` to force HuggingFace.

    Returns
    -------
    datasets.Dataset or datasets.IterableDataset
    """
    import os

    from datasets import load_dataset

    local_dir = data_dir if data_dir is not None else LONGBENCH_LOCAL_DIR
    local_path = os.path.join(local_dir, f"{dataset_name}.jsonl")

    if not use_e_variant and os.path.isfile(local_path):
        log.info("Loading LongBench dataset from local file: %s", local_path)
        ds = load_dataset("json", data_files=local_path, split="train")
        return ds

    config_name = f"{dataset_name}_e" if use_e_variant else dataset_name
    log.info(
        "Local file not found; loading LongBench dataset from HuggingFace: %s (config=%s)",
        dataset_name,
        config_name,
    )
    ds = load_dataset(
        "THUDM/LongBench",
        config_name,
        split="test",
        streaming=streaming,
    )
    return ds


def get_dataset_list(
    include_chinese: bool = False,
    use_e_variants: bool = False,
    custom_list: Optional[List[str]] = None,
) -> List[str]:
    """Return the list of dataset names to evaluate on.

    Parameters
    ----------
    include_chinese : bool
        If True, include the 5 Chinese datasets.
    use_e_variants : bool
        If True, return the 13 LongBench-E dataset names instead.
    custom_list : list of str, optional
        If provided, use this list directly (validated against known names).

    Returns
    -------
    list of str
    """
    if custom_list is not None:
        all_known = set(LONGBENCH_EN_DATASETS + LONGBENCH_ZH_DATASETS)
        unknown = [d for d in custom_list if d not in all_known]
        if unknown:
            log.warning("Unknown dataset names (will attempt anyway): %s", unknown)
        return custom_list

    if use_e_variants:
        return list(LONGBENCH_E_DATASETS)

    datasets = list(LONGBENCH_EN_DATASETS)
    if include_chinese:
        datasets.extend(LONGBENCH_ZH_DATASETS)
    return datasets
