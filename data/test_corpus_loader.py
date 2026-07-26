"""Tests for data/corpus_loader.py — deterministic sampling, no GPU required."""

from __future__ import annotations

import json

import pytest

from data.corpus_loader import CorpusLoader


class TestCorpusLoaderValidation:
    """Test input validation without loading data."""

    def test_rejects_unknown_dataset(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dataset"):
            CorpusLoader("imagenet")

    def test_accepts_wikitext(self) -> None:
        loader = CorpusLoader("wikitext-103")
        assert loader.dataset == "wikitext-103"

    def test_accepts_pg19(self) -> None:
        loader = CorpusLoader("pg19")
        assert loader.dataset == "pg19"


class TestArticleSplitting:
    """Test the wikitext article splitting logic on synthetic data."""

    def test_split_into_articles(self) -> None:
        text = (
            " = Article One =\nSome content here.\n\n"
            " = Article Two =\nMore content.\n\n"
            " = Article Three =\nFinal piece."
        )
        articles = CorpusLoader._split_into_articles(text)
        assert len(articles) >= 2  # At least two articles from the splits

    def test_empty_articles_dropped(self) -> None:
        text = " = Title =\n\n\n\n = Another =\nContent"
        articles = CorpusLoader._split_into_articles(text)
        for article in articles:
            assert article.strip() != ""


class TestDeterministicSampling:
    """Test that sampling is deterministic given a fixed seed.

    Uses a mock corpus (monkey-patched) to avoid dataset downloads.
    """

    @pytest.fixture
    def mock_loader(self) -> CorpusLoader:
        """Create a loader with pre-loaded mock articles."""
        loader = CorpusLoader("wikitext-103")
        loader._articles = [f"Article {i}: content {i * 7}" for i in range(100)]
        return loader

    def test_same_seed_same_result(self, mock_loader: CorpusLoader) -> None:
        sample_a = mock_loader.sample_articles(10, seed=42)
        sample_b = mock_loader.sample_articles(10, seed=42)
        assert sample_a == sample_b

    def test_different_seed_different_result(self, mock_loader: CorpusLoader) -> None:
        sample_a = mock_loader.sample_articles(10, seed=42)
        sample_b = mock_loader.sample_articles(10, seed=99)
        assert sample_a != sample_b

    def test_sample_preserves_order(self, mock_loader: CorpusLoader) -> None:
        """Samples should be in sorted (ascending) index order."""
        sample = mock_loader.sample_articles(10, seed=42)
        # Each article starts with "Article N:" — extract N
        indices = [int(s.split(":")[0].split(" ")[1]) for s in sample]
        assert indices == sorted(indices)

    def test_sample_too_many_raises(self, mock_loader: CorpusLoader) -> None:
        with pytest.raises(ValueError, match="Requested 200"):
            mock_loader.sample_articles(200, seed=42)


class TestGetArticle:
    """Test article retrieval by index."""

    @pytest.fixture
    def mock_loader(self) -> CorpusLoader:
        loader = CorpusLoader("wikitext-103")
        loader._articles = ["zero", "one", "two"]
        return loader

    def test_get_valid_article(self, mock_loader: CorpusLoader) -> None:
        assert mock_loader.get_article(0) == "zero"
        assert mock_loader.get_article(2) == "two"

    def test_get_out_of_range(self, mock_loader: CorpusLoader) -> None:
        with pytest.raises(IndexError):
            mock_loader.get_article(999)

    def test_get_negative_index(self, mock_loader: CorpusLoader) -> None:
        with pytest.raises(IndexError):
            mock_loader.get_article(-1)

    def test_num_articles(self, mock_loader: CorpusLoader) -> None:
        assert mock_loader.num_articles() == 3


class TestLocalCorpus:
    """A corpus the user supplies as a file or directory (the Kaggle path).

    Kaggle has no hub access and mounts the corpus at /kaggle/input/<name>, so
    `parity.dataset` must accept a path as well as a built-in dataset name.
    """

    def _jsonl(self, tmp_path, rows, name="docs.jsonl"):
        p = tmp_path / name
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return p

    def test_jsonl_with_text_field(self, tmp_path):
        p = self._jsonl(tmp_path, [{"text": "alpha"}, {"text": "beta"}])
        loader = CorpusLoader(str(p))
        assert loader.load() == ["alpha", "beta"]          # file order preserved
        assert loader.num_articles() == 2

    def test_local_prefix_is_accepted(self, tmp_path):
        p = self._jsonl(tmp_path, [{"text": "alpha"}])
        assert CorpusLoader(f"local:{p}").load() == ["alpha"]

    def test_alternate_field_names_and_bare_strings(self, tmp_path):
        p = self._jsonl(tmp_path, [{"content": "a"}, {"document": "b"}, "c"])
        assert CorpusLoader(str(p)).load() == ["a", "b", "c"]

    def test_single_string_field_is_used_as_fallback(self, tmp_path):
        p = self._jsonl(tmp_path, [{"weird_name": "a", "n_tokens": 3}])
        assert CorpusLoader(str(p)).load() == ["a"]

    def test_ambiguous_record_raises_with_keys(self, tmp_path):
        p = self._jsonl(tmp_path, [{"foo": "a", "bar": "b"}])
        with pytest.raises(ValueError, match="cannot tell which field"):
            CorpusLoader(str(p)).load()

    def test_explicit_text_field_wins(self, tmp_path):
        p = self._jsonl(tmp_path, [{"text": "ignored", "raw": "wanted"}])
        assert CorpusLoader(str(p), text_field="raw").load() == ["wanted"]

    def test_missing_explicit_field_raises(self, tmp_path):
        p = self._jsonl(tmp_path, [{"text": "a"}])
        with pytest.raises(ValueError, match="text_field"):
            CorpusLoader(str(p), text_field="nope").load()

    def test_json_array_and_wrapper(self, tmp_path):
        flat = tmp_path / "flat.json"
        flat.write_text(json.dumps(["a", "b"]), encoding="utf-8")
        assert CorpusLoader(str(flat)).load() == ["a", "b"]
        wrapped = tmp_path / "wrapped.json"
        wrapped.write_text(json.dumps({"data": [{"text": "a"}]}), encoding="utf-8")
        assert CorpusLoader(str(wrapped)).load() == ["a"]

    def test_directory_is_walked_in_sorted_order(self, tmp_path):
        d = tmp_path / "corpus"
        (d / "nested").mkdir(parents=True)
        (d / "b.txt").write_text("second", encoding="utf-8")
        (d / "a.txt").write_text("first", encoding="utf-8")
        (d / "nested" / "c.jsonl").write_text(
            json.dumps({"text": "third"}), encoding="utf-8")
        assert CorpusLoader(str(d)).load() == ["first", "second", "third"]

    def test_blank_and_whitespace_articles_are_dropped(self, tmp_path):
        p = self._jsonl(tmp_path, [{"text": "  a  "}, {"text": "   "}, {"text": ""}])
        assert CorpusLoader(str(p)).load() == ["a"]

    def test_empty_corpus_raises(self, tmp_path):
        p = self._jsonl(tmp_path, [{"text": "  "}])
        with pytest.raises(ValueError, match="no non-empty articles"):
            CorpusLoader(str(p)).load()

    def test_empty_directory_raises(self, tmp_path):
        d = tmp_path / "empty"; d.mkdir()
        with pytest.raises(ValueError, match="No .*files found"):
            CorpusLoader(str(d)).load()

    def test_malformed_jsonl_names_the_line(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"text": "ok"}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
            CorpusLoader(str(p)).load()

    def test_unknown_name_that_is_not_a_path_still_raises(self):
        with pytest.raises(ValueError, match="Unsupported dataset"):
            CorpusLoader("wikitext-999")

    def test_slug_is_filename_safe(self, tmp_path):
        p = self._jsonl(tmp_path, [{"text": "a"}], name="my docs v2.jsonl")
        assert CorpusLoader(str(p)).slug == "my-docs-v2"
        assert CorpusLoader("wikitext-103").slug == "wikitext-103"

    def test_deterministic_sampling_on_a_local_corpus(self, tmp_path):
        p = self._jsonl(tmp_path, [{"text": f"doc{i}"} for i in range(20)])
        loader = CorpusLoader(str(p))
        assert loader.sample_articles(5, seed=7) == loader.sample_articles(5, seed=7)
