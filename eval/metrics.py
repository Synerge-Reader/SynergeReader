import re
import string
from typing import List


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize(text: str) -> List[str]:
    return normalize(text).split()


def exact_match(predicted: str, expected: str) -> bool:
    return normalize(predicted) == normalize(expected)


def f1_score(predicted: str, expected: str) -> float:
    pred_tokens = tokenize(predicted)
    exp_tokens = tokenize(expected)

    if not pred_tokens or not exp_tokens:
        return 1.0 if pred_tokens == exp_tokens else 0.0

    pred_counts: dict = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    exp_counts: dict = {}
    for t in exp_tokens:
        exp_counts[t] = exp_counts.get(t, 0) + 1

    common = sum(min(pred_counts.get(t, 0), exp_counts[t]) for t in exp_counts)

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)
