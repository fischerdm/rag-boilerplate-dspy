"""Step 1: zero-shot baseline with dspy.Predict, then look at the actual prompt."""

import dspy

from common import TopicOf, evaluator, setup

trainset, devset = setup()

baseline = dspy.Predict(TopicOf)

# One call by hand, to see what a Prediction looks like.
pred = baseline(text=devset[0].text)
print(f"gold={devset[0].topic!r}  pred={pred.topic!r}")

# The prompt DSPy built from the signature, and the raw LM answer.
dspy.inspect_history(n=1)

evaluator(devset)(baseline)
