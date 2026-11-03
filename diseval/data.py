"""Load the NCBI Disease sentences (Hugging Face parquet) and decode the gold mentions."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# Tag ids in the Hugging Face copy. Confirm against the dataset card.
O, B, I = 0, 1, 2


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    gold: list[str]


def spans_from_tags(tags: list[int]) -> list[tuple[int, int]]:
    """Turn a BIO tag sequence into (start, end) token spans, end exclusive."""
    spans: list[tuple[int, int]] = []
    start = None
    for i, tag in enumerate(tags):
        if tag == B:
            if start is not None:
                spans.append((start, i))
            start = i
        elif tag == I:
            if start is None:  # a stray "inside" tag starts a span
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(tags)))
    return spans


def detokenize(tokens: list[str]) -> str:
    """Join tokens back into readable text (the corpus stores pre-split words)."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text


def load_split(path: str, limit: int | None = None) -> list[Example]:
    """Read a parquet file with columns: id, tokens, ner_tags."""
    df = pd.read_parquet(path)
    if limit:
        df = df.head(limit)
    examples = []
    for row in df.itertuples():
        tokens = [str(t) for t in row.tokens]
        tags = [int(t) for t in row.ner_tags]
        gold = [detokenize(tokens[s:e]) for s, e in spans_from_tags(tags)]
        examples.append(Example(id=str(row.id), text=detokenize(tokens), gold=gold))
    return examples
