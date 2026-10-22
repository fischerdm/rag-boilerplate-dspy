"""Shared pieces for all steps: data, signature, metric, LM setup.

Two tasks:
- topic classification (01-03, results.ipynb): single-label articles, predict the topic.
- label completion (04-05): given an article and its labels, add the missing ones.
"""

import os
import random
import re
from typing import Literal

import dspy
import nltk
from dotenv import load_dotenv

nltk.download("reuters", quiet=True)
from nltk.corpus import reuters  # noqa: E402  (needs the download above)

TOPICS = ["earn", "acq", "crude", "money-fx", "grain", "trade", "interest"]


def load(split: str, per_topic: int, seed: int = 0) -> list[dspy.Example]:
    """Load single-label articles, `per_topic` of each topic (fewer if the topic is rare).

    Sampling per topic instead of uniformly: a uniform sample is ~80% earn/acq,
    which would make accuracy mostly about those two classes.
    """
    rng = random.Random(seed)
    examples = []
    for topic in TOPICS:
        ids = [f for f in reuters.fileids(topic)
               if f.startswith(split) and reuters.categories(f) == [topic]]
        rng.shuffle(ids)
        examples += [dspy.Example(text=reuters.raw(f)[:1500], topic=topic).with_inputs("text")
                     for f in ids[:per_topic]]
    rng.shuffle(examples)
    return examples


class TopicOf(dspy.Signature):
    """Classify a 1987 Reuters newswire article into its topic."""

    text: str = dspy.InputField()
    # Same as Literal["earn", "acq", ...], the style used in the DSPy tutorials.
    # Valid at runtime; type checkers only accept values written out literally.
    topic: Literal[tuple(TOPICS)] = dspy.OutputField()  # pyright: ignore[reportInvalidTypeForm]


def metric(example, pred, trace=None):
    return example.topic == pred.topic


def configure_lm() -> None:
    load_dotenv()
    dspy.configure(lm=dspy.LM(os.getenv("DSPY_MODEL", "openai/gpt-4o-mini")))


def setup() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Configure the LM and return (trainset, devset)."""
    configure_lm()
    # grain has only 10 single-label test articles, so devset has 100 examples.
    return load("training", per_topic=8), load("test", per_topic=15)


# --- Label completion -------------------------------------------------------

# 61 articles have no body, only a date (and sometimes a time) printed twice.
_DATE = re.compile(r"\d{1,2}-[A-Z]{3}-\d{4}( [\d:.]+)?")


def has_text(fileid: str) -> bool:
    return _DATE.sub("", reuters.raw(fileid)).strip() != ""


def labels_of(fileid: str) -> list[str]:
    """The article's labels among TOPICS. Other labels (wheat, corn, dlr, ...) are out of scope."""
    return [c for c in reuters.categories(fileid) if c in TOPICS]


def load_completion(split: str, single_per_topic: int, max_multi: int | None = None,
                    seed: int = 0) -> list[dspy.Example]:
    """Examples for label completion, built from articles with text and at least one label in TOPICS.

    - Multi-label articles: one label is hidden, the model should add it back (recall).
    - Single-label articles: nothing is known to be missing. These are the articles we
      actually want to extend; here they show how often the model adds a label at all.

    Each example has `existing_topics` (input), `missing_topics` (what should be added)
    and `labels` (all labels, for analysis).
    """
    rng = random.Random(seed)
    ids = [f for f in reuters.fileids() if f.startswith(split) and labels_of(f) and has_text(f)]

    def example(f: str, existing: list[str], missing: list[str]) -> dspy.Example:
        return dspy.Example(text=reuters.raw(f)[:1500], existing_topics=existing,
                            missing_topics=missing, labels=labels_of(f),
                            fileid=f).with_inputs("text", "existing_topics")

    examples = []
    multi = [f for f in ids if len(labels_of(f)) > 1]
    rng.shuffle(multi)
    for f in multi[:max_multi]:
        labels = labels_of(f)
        hidden = rng.choice(labels)
        examples.append(example(f, [t for t in labels if t != hidden], [hidden]))

    for topic in TOPICS:
        single = [f for f in ids if labels_of(f) == [topic]]
        rng.shuffle(single)
        examples += [example(f, [topic], []) for f in single[:single_per_topic]]

    rng.shuffle(examples)
    return examples


class AddTopics(dspy.Signature):
    """A 1987 Reuters newswire article is already labeled with some topics.
    List the other topics that also clearly apply. Return an empty list if none do."""

    text: str = dspy.InputField()
    existing_topics: list[str] = dspy.InputField()
    additional_topics: list[Literal[tuple(TOPICS)]] = dspy.OutputField()  # pyright: ignore[reportInvalidTypeForm]


def added(example, pred) -> set[str]:
    """Topics the model added; repeating an existing topic does not count."""
    return set(pred.additional_topics) - set(example.existing_topics)


def completion_metric(example, pred, trace=None):
    """Exact match: the hidden topic is added back and nothing else is added.

    Too strict as a quality measure: an addition to a single-label article counts as a
    failure, although it may be a genuinely missing label. dspy.Evaluate and optimizers
    need one number per example, so it is used there; read `completion_scores` instead.
    """
    return added(example, pred) == set(example.missing_topics)


def completion_scores(result) -> dict[str, float]:
    """The two numbers to look at, from a `dspy.Evaluate` result.

    - recall: share of multi-label articles where the hidden topic is added back.
    - addition_rate: share of single-label articles that get any topic added. Neither
      right nor wrong by itself: ~0 means the model never extends, ~1 means it always does.
      Whether the additions are justified needs a manual look.
    """
    multi = [(ex, p) for ex, p, _ in result.results if ex.missing_topics]
    single = [(ex, p) for ex, p, _ in result.results if not ex.missing_topics]
    return {
        "recall": sum(ex.missing_topics[0] in added(ex, p) for ex, p in multi) / len(multi),
        "addition_rate": sum(bool(added(ex, p)) for ex, p in single) / len(single),
    }


def setup_completion() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Configure the LM and return (trainset, devset) for label completion."""
    configure_lm()
    # devset: all 68 multi-label test articles + 10 single-label articles per topic.
    return (load_completion("training", single_per_topic=4, max_multi=30),
            load_completion("test", single_per_topic=10))


def print_additions(result) -> None:
    """The topics added to single-label articles: the candidates for manual review."""
    for ex, p, _ in result.results:
        if not ex.missing_topics and added(ex, p):
            headline = ex.text.strip().splitlines()[0][:70]
            print(f"{ex.fileid:12} {ex.existing_topics[0]:9} +{sorted(added(ex, p))}  {headline}")


def completion_evaluator(devset: list[dspy.Example], display_progress: bool = True) -> dspy.Evaluate:
    """Runs the program on devset; its printed score is the strict metric, see `completion_scores`."""
    return dspy.Evaluate(devset=devset, metric=completion_metric, num_threads=8,
                         display_progress=display_progress)


def evaluator(devset: list[dspy.Example], display_progress: bool = True) -> dspy.Evaluate:
    return dspy.Evaluate(devset=devset, metric=metric, num_threads=8,
                         display_progress=display_progress)
