"""Step 3: let an optimizer improve the program using the trainset, then save it.

BootstrapFewShot runs the program on trainset examples and keeps the traces
where `metric` passes as few-shot demos. MIPROv2 additionally rewrites the
instruction (more LM calls, uncomment to try).
"""

from pathlib import Path

import dspy

from common import TopicOf, evaluator, metric, setup

trainset, devset = setup()
evaluate = evaluator(devset)

optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=4)
# optimizer = dspy.MIPROv2(metric=metric, auto="light", num_threads=8)

optimized = optimizer.compile(dspy.ChainOfThought(TopicOf), trainset=trainset)

evaluate(optimized)

# Look at what the optimizer chose: instruction + demos now live in the program.
dspy.inspect_history(n=1)

Path("artifacts").mkdir(exist_ok=True)
optimized.save("artifacts/optimized_cot.json")
# Reload later with:
#   program = dspy.ChainOfThought(TopicOf); program.load("artifacts/optimized_cot.json")
