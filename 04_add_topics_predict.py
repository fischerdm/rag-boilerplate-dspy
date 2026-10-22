"""Step 4: label completion with dspy.Predict.

Given an article and its existing topics, the model lists the topics that also apply.
See `load_completion` and `completion_scores` in common.py for how this is evaluated.
"""

import dspy

from common import AddTopics, completion_evaluator, completion_scores, print_additions, setup_completion

trainset, devset = setup_completion()

predict = dspy.Predict(AddTopics)

# One call by hand: a list-valued output field.
ex = devset[0]
pred = predict(text=ex.text, existing_topics=ex.existing_topics)
print(f"existing={ex.existing_topics}  missing={ex.missing_topics}  pred={pred.additional_topics}")

dspy.inspect_history(n=1)

result = completion_evaluator(devset)(predict)
print(completion_scores(result))

print("\nAdded to single-label articles:")
print_additions(result)
