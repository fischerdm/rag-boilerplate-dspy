"""Step 2: same signature, different module. ChainOfThought adds a `reasoning` output."""

import dspy

from common import TopicOf, evaluator, setup

trainset, devset = setup()

cot = dspy.ChainOfThought(TopicOf)

pred = cot(text=devset[0].text)
print(f"gold={devset[0].topic!r}  pred={pred.topic!r}")
print("reasoning:", pred.reasoning)

evaluator(devset)(cot)
