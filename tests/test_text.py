from __future__ import annotations

import pytest

from rk.text import (
    content_stems,
    extract_acronyms,
    normalize_title,
    reconstruct_abstract,
    stem,
    tokenize,
)


def test_reconstruct_abstract_orders_words_by_position() -> None:
    inverted = {"retrieval": [1, 4], "sparse": [3], "Dense": [0], "beats": [2]}
    assert reconstruct_abstract(inverted) == "Dense retrieval beats sparse retrieval"


def test_reconstruct_abstract_handles_missing_and_gapped_indexes() -> None:
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""
    assert reconstruct_abstract({"end.": [7], "Start": [0]}) == "Start end."


def test_extract_acronyms_finds_named_artefacts() -> None:
    text = (
        "We fine-tune BERT and GPT-4 on ImageNet with LoRA, compare CNNs, word2vec "
        "and BM25, and test Llama-2 at Large-Scale in Phase II, ranked 1st."
    )
    expected = ["BERT", "BM25", "CNN", "GPT", "GPT-4", "ImageNet", "Llama-2", "LoRA", "word2vec"]
    assert extract_acronyms(text) == expected


def test_extract_acronyms_ignores_all_caps_titles() -> None:
    title = "DEEP LEARNING FOR MEDICAL IMAGE SEGMENTATION WITH GPT-4"
    assert extract_acronyms(title) == ["GPT-4"]


def test_extract_acronyms_on_plain_prose_is_empty() -> None:
    assert extract_acronyms("A Study of Graph Neural Networks for Molecules.") == []


def test_tokenize_lowercases_folds_accents_and_splits_punctuation() -> None:
    tokens = tokenize("Self-Supervised Learning, naïve 3D models!")
    assert tokens == ["self", "supervised", "learning", "naive", "3d", "models"]


@pytest.mark.parametrize(
    "family",
    [
        ["segment", "segments", "segmented", "segmenting", "segmentation"],
        ["optimize", "optimized", "optimizing", "optimization", "optimizer", "optimisation"],
        ["embed", "embedding", "embeddings", "embedded"],
        ["study", "studies", "studied", "studying"],
        ["process", "processes", "processing", "processed"],
        ["cluster", "clusters", "clustering", "clustered"],
        ["image", "images", "imaging"],
        ["model", "models", "modeling", "modelling"],
    ],
)
def test_stem_conflates_inflected_forms(family: list[str]) -> None:
    assert len({stem(w) for w in family}) == 1, {w: stem(w) for w in family}


def test_stem_leaves_short_and_alphanumeric_tokens_alone() -> None:
    assert stem("map") == "map"
    assert stem("gpt4") == "gpt4"
    assert stem("generation") != stem("generalization")


def test_content_stems_drop_stopwords_numbers_and_optionally_generic_words() -> None:
    text = "We propose a novel transformer for segmentation in 2024"
    assert content_stems(text) == ["propos", "novel", "transform", "seg"]
    assert content_stems(text, drop_generic=True) == ["transform", "seg"]


def test_normalize_title_ignores_markup_case_punctuation_and_accents() -> None:
    a = normalize_title("Pre-Training <i>BERT</i>: A Naïve Study &amp; More.")
    b = normalize_title("pretraining BERT — a naive study & more")
    assert a == b == "pretrainingbertanaivestudymore"
    assert normalize_title(None) == normalize_title("") == ""
