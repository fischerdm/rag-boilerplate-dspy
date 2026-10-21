"""Shared pieces for all steps: data, signature, metric, LM setup."""

import os
import random
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
    topic: Literal[tuple(TOPICS)] = dspy.OutputField()


def metric(example, pred, trace=None):
    return example.topic == pred.topic


def setup() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Configure the LM and return (trainset, devset)."""
    load_dotenv()
    dspy.configure(lm=dspy.LM(os.getenv("DSPY_MODEL", "openai/gpt-4o-mini")))
    # grain has only 10 single-label test articles, so devset has 100 examples.
    return load("training", per_topic=8), load("test", per_topic=15)


def evaluator(devset: list[dspy.Example]) -> dspy.Evaluate:
    return dspy.Evaluate(devset=devset, metric=metric, num_threads=8, display_progress=True)
