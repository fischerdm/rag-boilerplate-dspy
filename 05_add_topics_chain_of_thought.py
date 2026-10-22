"""Step 5: label completion with dspy.ChainOfThought (adds a `reasoning` output)."""

import dspy

from common import AddTopics, completion_evaluator, completion_scores, print_additions, setup_completion

trainset, devset = setup_completion()

cot = dspy.ChainOfThought(AddTopics)

ex = devset[0]
pred = cot(text=ex.text, existing_topics=ex.existing_topics)
print(f"existing={ex.existing_topics}  missing={ex.missing_topics}  pred={pred.additional_topics}")
print("reasoning:", pred.reasoning)

result = completion_evaluator(devset)(cot)
print(completion_scores(result))

print("\nAdded to single-label articles:")
print_additions(result)
