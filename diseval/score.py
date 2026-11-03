"""Score extracted mentions against the gold annotations.

Usage:
    python -m diseval.score results/val_raw.jsonl --errors results/val_errors.csv

Two scores are reported, both micro-averaged over all sentences:
  strict   a prediction counts only if it equals a gold mention after normalisation
  lenient  also counts a prediction that contains, or is contained in, a gold mention
           (for example "adenomatous polyposis coli" vs "adenomatous polyposis coli tumour")
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass

import pandas as pd


def normalise(s: str) -> str:
    """Lowercase and collapse punctuation and spacing so 'BRCA1 - related' equals 'BRCA1-related'."""
    return re.sub(r"[\W_]+", " ", s.lower()).strip()


def _overlaps(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (f" {b} " in f" {a} " or f" {a} " in f" {b} ")


@dataclass
class Match:
    tp: int
    fp: list[str]
    fn: list[str]


def match_sentence(gold: list[str], pred: list[str], lenient: bool = False) -> Match:
    """One-to-one matching of predictions to gold mentions. Exact matches are made first."""
    gn = [normalise(g) for g in gold]
    pn = [normalise(p) for p in pred]
    g_used = [False] * len(gold)
    p_used = [False] * len(pred)

    def run(same) -> None:
        for i, g in enumerate(gn):
            if g_used[i]:
                continue
            for j, p in enumerate(pn):
                if not p_used[j] and same(g, p):
                    g_used[i] = p_used[j] = True
                    break

    run(lambda g, p: bool(g) and g == p)
    if lenient:
        run(_overlaps)
    return Match(
        tp=sum(g_used),
        fp=[pred[j] for j, used in enumerate(p_used) if not used],
        fn=[gold[i] for i, used in enumerate(g_used) if not used],
    )


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def evaluate(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Return the strict and lenient scores plus a list of error records."""
    totals = {m: [0, 0, 0] for m in ("strict", "lenient")}
    errors: list[dict] = []
    for row in rows:
        for name in totals:
            m = match_sentence(row["gold"], row["pred"], lenient=(name == "lenient"))
            totals[name][0] += m.tp
            totals[name][1] += len(m.fp)
            totals[name][2] += len(m.fn)
            if name == "strict":
                errors += [{"id": row["id"], "kind": "false_positive", "mention": x, "text": row["text"]} for x in m.fp]
                errors += [{"id": row["id"], "kind": "missed", "mention": x, "text": row["text"]} for x in m.fn]
    scores = {name: prf(*vals) for name, vals in totals.items()}
    scores["sentences"] = len(rows)
    scores["failed_calls"] = sum(1 for r in rows if r.get("error"))
    scores["sentences_with_disease"] = sum(1 for r in rows if r["gold"])
    return scores, errors


def read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="JSONL file written by diseval.extract")
    ap.add_argument("--errors", help="CSV file to write the misses and false positives to")
    args = ap.parse_args()

    scores, errors = evaluate(read_rows(args.raw))
    print(f"Sentences: {scores['sentences']} ({scores['sentences_with_disease']} with at least one disease)")
    print(f"Failed calls: {scores['failed_calls']}\n")
    print("| Match   | Precision | Recall | F1    | TP  | FP  | FN  |")
    print("|---------|-----------|--------|-------|-----|-----|-----|")
    for name in ("strict", "lenient"):
        s = scores[name]
        print(f"| {name:<7} | {s['precision']:.3f}     | {s['recall']:.3f}  | {s['f1']:.3f} | {s['tp']:<3} | {s['fp']:<3} | {s['fn']:<3} |")
    if args.errors:
        pd.DataFrame(errors).to_csv(args.errors, index=False)
        print(f"\nWrote {len(errors)} errors to {args.errors}")


if __name__ == "__main__":
    main()
