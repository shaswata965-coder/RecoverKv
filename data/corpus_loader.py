"""Deterministic corpus loaders for wikitext-103, PG19, and local corpora.

Each loader returns articles as a list of strings. Sampling is deterministic
given a seed — the same ``(dataset, article_id, seed)`` triple always
returns the same article text.

``dataset`` is either a built-in name (``"wikitext-103"`` / ``"pg19"``, pulled
from the HuggingFace hub) or a **filesystem path** to a corpus you supply —
which is what makes the parity suites runnable on Kaggle, where the hub is
unreachable and the corpus arrives as an attached dataset directory:

    parity.dataset: /kaggle/input/my-corpus/docs.jsonl     # one doc per line
    parity.dataset: /kaggle/input/my-corpus                # dir of .txt/.jsonl
    parity.dataset: local:./data/mine.json                 # explicit prefix

Accepted layouts (order is always deterministic — jsonl/json keep file order,
directories are walked in sorted path order):

    <file>.jsonl / .ndjson   one JSON object (or bare string) per line
    <file>.json              a JSON array of objects or strings
    <file>.txt               exactly one article
    <dir>/                   every .txt / .jsonl / .ndjson / .json inside,
                             recursively, sorted by path

For object records the text field is auto-detected from ``_TEXT_FIELDS`` in
priority order; if none matches and the record has exactly one string-valued
field, that field is used. Anything else raises with the keys it actually saw,
rather than silently picking a column.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from utils.logger import get_logger

log = get_logger(__name__)

_LOCAL_PREFIX = "local:"
_SUFFIXES = (".jsonl", ".ndjson", ".json", ".txt")


class CorpusLoader:
    """Load and sample articles from wikitext-103, PG19, or a local corpus.

    Parameters
    ----------
    dataset : str
        ``"wikitext-103"``, ``"pg19"``, or a path to a local corpus file or
        directory (optionally prefixed ``local:``).
    cache_dir : str, optional
        HuggingFace datasets cache directory (built-in datasets only).
    text_field : str, optional
        Force a specific JSON field instead of auto-detecting.
    """

    _SUPPORTED_DATASETS = {"wikitext-103", "pg19"}
    _TEXT_FIELDS = ("text", "content", "document", "article", "body",
                    "input", "prompt", "context")

    def __init__(self, dataset: str, cache_dir: Optional[str] = None,
                 text_field: Optional[str] = None) -> None:
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.text_field = text_field
        self._articles: Optional[List[str]] = None

        spec = dataset[len(_LOCAL_PREFIX):] if dataset.startswith(_LOCAL_PREFIX) \
            else dataset
        self.local_path: Optional[Path] = None
        if dataset not in self._SUPPORTED_DATASETS:
            candidate = Path(spec).expanduser()
            if candidate.exists():
                self.local_path = candidate
            else:
                raise ValueError(
                    f"Unsupported dataset: {dataset!r}. Choose from "
                    f"{sorted(self._SUPPORTED_DATASETS)}, or give a path to a "
                    f"local corpus file/directory (looked for {candidate})."
                )

    @property
    def slug(self) -> str:
        """Filesystem-safe short name, for output filenames.

        Built-in datasets keep their name; a local corpus becomes its stem
        (``/kaggle/input/my-corpus/docs.jsonl`` → ``docs``), so the parity
        runners' default npz names stay valid paths.
        """
        raw = self.local_path.stem if self.local_path is not None else self.dataset
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")
        return slug or "corpus"

    def _load_wikitext103(self) -> List[str]:
        """Load wikitext-103 and split into articles."""
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            split="test",
            cache_dir=self.cache_dir,
        )
        # Concatenate all text, then split on article boundaries (double newlines
        # following a title pattern "= Title =" at the start of a line)
        full_text = "\n".join(row["text"] for row in ds)
        articles = self._split_into_articles(full_text)
        log.info("Loaded wikitext-103: %d articles", len(articles))
        return articles

    def _load_pg19(self) -> List[str]:
        """Load PG19 test split and return individual books."""
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset(
            "deepmind/pg19",
            split="test",
            cache_dir=self.cache_dir,
        )
        articles = [row["text"] for row in ds]
        log.info("Loaded pg19: %d articles", len(articles))
        return articles

    # ------------------------------------------------------------------
    # Local corpora
    # ------------------------------------------------------------------

    def _extract_text(self, record, origin: str) -> Optional[str]:
        """Pull the article text out of one JSON record."""
        if isinstance(record, str):
            return record
        if not isinstance(record, dict):
            raise ValueError(
                f"{origin}: expected a string or object per record, got "
                f"{type(record).__name__}")
        if self.text_field is not None:
            if self.text_field not in record:
                raise ValueError(
                    f"{origin}: text_field {self.text_field!r} not in record "
                    f"(keys: {sorted(record)})")
            return str(record[self.text_field])
        for field in self._TEXT_FIELDS:
            if isinstance(record.get(field), str):
                return record[field]
        strings = [k for k, v in record.items() if isinstance(v, str)]
        if len(strings) == 1:
            return record[strings[0]]
        raise ValueError(
            f"{origin}: cannot tell which field holds the article text "
            f"(keys: {sorted(record)}). Pass text_field explicitly.")

    def _load_local_file(self, path: Path) -> List[str]:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return [path.read_text(encoding="utf-8", errors="replace")]
        if suffix in (".jsonl", ".ndjson"):
            out = []
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"{path}:{lineno}: invalid JSON ({e})") from e
                    text = self._extract_text(record, f"{path}:{lineno}")
                    if text:
                        out.append(text)
            return out
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(payload, dict):
                # A single {"data": [...]} style wrapper, or one record.
                for key in ("data", "records", "articles", "documents"):
                    if isinstance(payload.get(key), list):
                        payload = payload[key]
                        break
                else:
                    payload = [payload]
            if not isinstance(payload, list):
                raise ValueError(f"{path}: expected a JSON array of records")
            texts = [self._extract_text(r, str(path)) for r in payload]
            return [t for t in texts if t]
        raise ValueError(f"{path}: unsupported corpus file type {suffix!r}")

    def _load_local(self) -> List[str]:
        path = self.local_path
        assert path is not None
        if path.is_dir():
            files = sorted(
                p for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in _SUFFIXES
            )
            if not files:
                raise ValueError(
                    f"No {'/'.join(_SUFFIXES)} files found under {path}")
        else:
            files = [path]
        articles: List[str] = []
        for f in files:
            articles.extend(self._load_local_file(f))
        articles = [a.strip() for a in articles if a and a.strip()]
        if not articles:
            raise ValueError(f"Local corpus {path} produced no non-empty articles")
        log.info("Loaded local corpus %s: %d article(s) from %d file(s)",
                 path, len(articles), len(files))
        return articles

    @staticmethod
    def _split_into_articles(text: str) -> List[str]:
        """Split raw wikitext into individual articles.

        Articles are delimited by lines starting with ``" = "`` (level-1
        headings in the wikitext markup).  Empty articles are dropped.
        """
        import re

        # Split on level-1 headings (single = on each side)
        parts = re.split(r"\n(?= = [^=])", text)
        articles = [p.strip() for p in parts if p.strip()]
        return articles

    def load(self) -> List[str]:
        """Load and cache the article list. Idempotent."""
        if self._articles is not None:
            return self._articles

        if self.local_path is not None:
            self._articles = self._load_local()
        elif self.dataset == "wikitext-103":
            self._articles = self._load_wikitext103()
        elif self.dataset == "pg19":
            self._articles = self._load_pg19()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        return self._articles

    def get_article(self, article_id: int) -> str:
        """Return the article at index *article_id*.

        Raises
        ------
        IndexError
            If *article_id* is out of range.
        """
        articles = self.load()
        if article_id < 0 or article_id >= len(articles):
            raise IndexError(
                f"article_id {article_id} out of range for "
                f"{self.dataset} ({len(articles)} articles)"
            )
        return articles[article_id]

    def num_articles(self) -> int:
        """Return the number of articles in the loaded corpus."""
        return len(self.load())

    def sample_articles(
        self, n: int, seed: int = 42
    ) -> List[str]:
        """Deterministically sample *n* articles using *seed*.

        Parameters
        ----------
        n : int
            Number of articles to sample.
        seed : int
            RNG seed for reproducible sampling.

        Returns
        -------
        list[str]
            Sampled article texts.
        """
        import random as _random

        articles = self.load()
        if n > len(articles):
            raise ValueError(
                f"Requested {n} articles but only {len(articles)} available"
            )

        rng = _random.Random(seed)
        indices = rng.sample(range(len(articles)), n)
        indices.sort()  # Deterministic ordering
        return [articles[i] for i in indices]
