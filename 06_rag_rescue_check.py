"""Step 6: would label completion have rescued the RAG cases the topic filter excluded?

In fastapi-rag-boilerplate, filtering by the topic inferred from the question excluded the
right article in 8 golden cases (docs/evaluation.md there, 'reviewed baseline'). For each,
ask whether the inferred topic applies to the article. 'Yes' means re-labeling the article
would have let it through the filter; 'no' means the question's topic was the problem.

Uses the full Reuters-21578 archive (as indexed there), not NLTK's labeled subset.
"""

import html
import re
import tarfile
import urllib.request
from pathlib import Path

import dspy

from common import configure_lm

ARCHIVE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz"
ARCHIVE_PATH = Path("data/reuters21578.tar.gz")

# case: (question, topic inferred from it, expected article NEWIDs)
CASES = {
    "q0017": ("What is Bank of British Columbia's revised shareholder distribution estimate after selling its assets?", "acq", [4019]),
    "q0024": ("What U.S. joint venture did Nippon Mining sign with Conoco Inc?", "acq", [17419]),
    "q0048": ("How much cargo passed through Abidjan port, and how did exports and imports change?", "trade", [10394]),
    "q0033": ("How much beef did Argentina export in Jan/Feb 1987?", "trade", [4022]),
    "q0014": ("What pact did California Micro Devices and Grumman's Tachonics sign in March?", "acq", [10151]),
    "q0015": ("Why is Cehave NV Veghel challenging the European Community grain co-responsibility levy?", "grain", [16146]),
    "q0010": ("What dividend and stock split did K mart announce on March 24?", "acq", [8846]),
    "q0027": ("What credit guarantees will Saudi Arabia accept from Petrobras?", "instal-debt", [5123, 5125, 5281]),
}

# Same wording as in AddTopics; instal-debt is not one of our 7 topics.
DEFINITIONS = {
    "acq": "mergers and acquisitions, buying or selling companies or stakes in them",
    "trade": "international trade between countries, e.g. trade balances, trade policy, tariffs, "
             "trade disputes (not a company buying or selling something)",
    "grain": "grain crops and grain markets, e.g. wheat, corn, rice, harvests, tenders",
    "instal-debt": "instalment debt and consumer credit",
}


def articles() -> dict[int, tuple[str, list[str]]]:
    """NEWID -> (title + body, topics), read straight from the archive."""
    if not ARCHIVE_PATH.exists():
        ARCHIVE_PATH.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(ARCHIVE_URL, ARCHIVE_PATH)

    def tag(name: str, text: str) -> str:
        match = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL)
        return html.unescape(match.group(1)).strip() if match else ""

    out = {}
    with tarfile.open(ARCHIVE_PATH) as archive:
        for member in archive.getmembers():
            if not member.name.endswith(".sgm"):
                continue
            sgm = archive.extractfile(member).read().decode("latin-1")
            for newid, art in re.findall(r'<REUTERS[^>]*NEWID="(\d+)"[^>]*>(.*?)</REUTERS>', sgm, re.DOTALL):
                out[int(newid)] = (f"{tag('TITLE', art)}\n{tag('BODY', art)}",
                                   re.findall(r"<D>(.*?)</D>", tag("TOPICS", art)))
    return out


class TopicApplies(dspy.Signature):
    """Does this Reuters topic apply to the 1987 Reuters newswire article?
    Use the Reuters meaning of the topic given in the definition, not the everyday sense of the word."""

    text: str = dspy.InputField()
    topic: str = dspy.InputField()
    definition: str = dspy.InputField()
    applies: bool = dspy.OutputField()


configure_lm()
check = dspy.ChainOfThought(TopicApplies)
docs = articles()

for case, (question, topic, ids) in CASES.items():
    for newid in ids:
        text, labels = docs[newid]
        pred = check(text=text[:1500], topic=topic, definition=DEFINITIONS[topic])
        print(f"{case} reuters-{newid}  labels={labels}  inferred={topic}  applies={pred.applies}")
        print(f"  Q: {question}")
        print(f"  {text.splitlines()[0]}")
        print(f"  reasoning: {pred.reasoning}\n")
