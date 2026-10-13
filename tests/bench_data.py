"""A small frozen corpus and hand-written proposals for the benchmark tests.

The three targets have realistic abstracts; the other documents are ordinary
ML abstracts, so that common research words get a low IDF. Each LEAKY entry
is the corresponding GOOD proposal with one change that trips one check.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

T1, T2, T3 = "W101", "W102", "W103"

FACETS = ("problem", "method", "domain", "claim")


def judge_reply(
    failed: Mapping[str, str] | None = None, *, verdict: str | None = None, reason: str = "ok"
) -> str:
    """A facet-by-facet judge answer in the prompts' JSON format.

    `failed` maps facets to the reason they don't match; the verdict defaults
    to "no" if any facet failed, else "yes".
    """
    failed = dict(failed or {})
    return json.dumps(
        {
            "facets": {f: {"paper": f"paper {f}", "proposal": f"proposal {f}"} for f in FACETS},
            "match": {
                f: {"match": "no", "reason": failed[f]}
                if f in failed
                else {"match": "yes", "reason": "same"}
                for f in FACETS
            },
            "verdict": verdict or ("no" if failed else "yes"),
            "reason": reason,
        }
    )

TITLES = {
    T1: "Sparse attention routing for long document summarisation",
    T2: "Calibrated reward models for preference learning",
    T3: "Graph networks for traffic forecasting under sensor failures",
}

ABSTRACTS = {
    T1: (
        "Transformer summarisers struggle with book-length inputs because full self-attention "
        "grows quadratically with input length. We introduce a routing layer that sends each "
        "token to a handful of learned clusters, so attention is only computed inside "
        "clusters. Trained end to end with the summarisation loss, the router keeps ROUGE "
        "within half a point of dense BERT-based baselines on arXiv and PubMed while cutting "
        "memory by a factor of eight. Ablations show that the number of clusters matters "
        "more than their size."
    ),
    T2: (
        "Reward models trained on human comparisons are often overconfident, which lets "
        "policies exploit them during RLHF. We fit one temperature per prompt category on "
        "held-out comparisons and show that the calibrated scores reduce reward hacking in "
        "three chat settings. The method adds no parameters to the policy and needs only one "
        "extra pass over the comparison data."
    ),
    T3: (
        "Road sensors fail often, and forecasting models trained on complete data degrade "
        "sharply when readings go missing. We train a graph network with random sensor "
        "dropout and a learned imputation head, so it forecasts traffic speed from whichever "
        "sensors are alive. On two city networks the model loses less than five percent "
        "accuracy with a third of sensors offline, where standard models lose a quarter."
    ),
}

OTHER_DOCS = [
    ("Deep models for image classification",
     ("We propose a deep model for image classification. Experiments on standard data show "
      "that the model improves accuracy over strong baselines while using less memory.")),
    ("Learning rate schedules revisited",
     ("We study learning rate schedules for training deep networks. Results show that simple "
      "schedules match tuned ones, and we report the memory and time each method needs.")),
    ("Efficient training of language models",
     ("We present a method that reduces the memory used to train language models on long "
      "inputs. The approach keeps quality close to the baseline while training faster.")),
    ("A benchmark for question answering",
     ("We introduce a benchmark of questions with human answers and evaluate several models. "
      "Results show a large gap between models and humans on questions that need reasoning.")),
    ("Robust speech recognition in noise",
     ("We train a speech model with added noise and show that it keeps accuracy when inputs "
      "are corrupted. The method needs no extra parameters and little extra training time.")),
    ("Contrastive learning for medical images",
     ("We apply contrastive learning to medical images. The learned features improve accuracy "
      "on three tasks, and the method needs only a small number of labelled images.")),
    ("Pruning neural networks at initialisation",
     ("We propose to remove weights before training. The pruned networks keep accuracy with "
      "a fraction of the parameters, and results hold across several model sizes.")),
    ("Data augmentation for text classification",
     ("We compare data augmentation methods for text classification. Simple word swaps help "
      "small data settings, while results on large data show little gain.")),
    ("Uncertainty estimates for regression",
     ("We train an ensemble of models and show that its uncertainty estimates are better "
      "calibrated than single models on several regression tasks with shifted data.")),
    ("Few-shot learning with prototypes",
     ("We learn a metric space in which each class is represented by the mean of its "
      "examples. The approach is simple and reaches strong accuracy with few examples.")),
]

GOOD = {
    T1: (
        "We plan to let a text condensation system pick, for every word, a few learned groups "
        "of related words and compare it only with members of those groups, so that very "
        "lengthy books can be condensed without running out of memory. The grouping would be "
        "fitted jointly with the condensation objective. We expect output quality close to "
        "exhaustive comparison while needing far less memory."
    ),
    T2: (
        "We propose to measure how often a learned judge of people's choices is too sure of "
        "itself, and to rescale its outputs separately for each kind of request using "
        "examples set aside for this purpose. A system optimised against the rescaled judge "
        "should then be less able to game it. This changes nothing in the system being "
        "optimised and costs a single additional sweep over the stored choices."
    ),
    T3: (
        "We aim to predict how fast vehicles move on urban roads even when many roadside "
        "detectors stop reporting. While fitting the predictor we would randomly silence "
        "detectors and add a component that guesses their absent values, so predictions rely "
        "on whichever detectors still work. We expect small losses when many detectors are "
        "silent, compared with large losses for conventional predictors."
    ),
}

# The first failing check is the recorded reason, so each of these trips one.
LEAKY = {
    "acronym_in_paraphrase": GOOD[T1].replace("a text condensation system", "an LLM"),
    "source_acronym": GOOD[T1].replace("exhaustive comparison", "exhaustive comparison on rouge"),
    "title_stem": GOOD[T1].replace("very lengthy books", "very long documents"),
    "idf_overlap": GOOD[T1].replace(
        "far less memory.",
        "far less memory. Ablations would vary the handful of clusters quadratically.",
    ),
    "content_jaccard": GOOD[T1].replace(
        "We expect output quality close to exhaustive comparison while needing far less memory.",
        "We show that each token passes through one layer fitted with the same loss, keeping "
        "quality within half a point while cutting memory, and that the number of groups "
        "matters more than their size.",
    ),
    "length": "We plan to condense very lengthy books without running out of memory.",
}


def corpus_records() -> list[dict[str, Any]]:
    """Corpus rows in the frozen format (only the fields the benchmark reads)."""
    rows = [
        {"id": t, "aliases": [], "title": TITLES[t], "abstract": ABSTRACTS[t]}
        for t in (T1, T2, T3)
    ]
    for n, (title, abstract) in enumerate(OTHER_DOCS, start=201):
        rows.append({"id": f"W{n}", "aliases": [], "title": title, "abstract": abstract})
    return rows


def target_rows() -> list[dict[str, Any]]:
    return [{"id": t, "aliases": [], "remapped_from": None} for t in (T1, T2, T3)]
