"""Ask an LLM to list the disease mentions in each sentence.

Usage (from the repo root):
    python -m diseval.extract --data data/validation.parquet --out results/val_raw.jsonl --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from pydantic import BaseModel, field_validator

from diseval.data import load_split

PROMPT_VERSION = "v1"
DEFAULT_MODEL = "claude-haiku-5-5"  # check the current model names in the Anthropic docs

SYSTEM = (
    "You extract disease mentions from biomedical text. "
    "Call the record_diseases tool with every disease, disorder or syndrome named in the text. "
    "Copy each mention exactly as it is written, including abbreviations, and list it once per "
    "mention in order of appearance. Do not include genes, proteins or chemicals unless they are "
    "part of a disease name. If the text names no disease, return an empty list."
)

TOOL = {
    "name": "record_diseases",
    "description": "Record the disease mentions found in the text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "diseases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Disease mentions, copied exactly as written in the text.",
            }
        },
        "required": ["diseases"],
    },
}


class DiseaseExtraction(BaseModel):
    diseases: list[str]

    @field_validator("diseases")
    @classmethod
    def drop_blank(cls, v: list[str]) -> list[str]:
        return [d.strip() for d in v if d.strip()]


def extract_one(client, model: str, text: str) -> list[str]:
    """Call the model once and return its validated list of mentions."""
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM,
        tools=[TOOL],
        tool_choice={"type": "tool", "name": TOOL["name"]},
        messages=[{"role": "user", "content": f"Text:\n{text}"}],
    )
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return DiseaseExtraction.model_validate(block.input).diseases
    raise ValueError("model did not call the tool")


def _done_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def main() -> None:
    import anthropic
    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="parquet file with id, tokens, ner_tags")
    ap.add_argument("--out", required=True, help="JSONL file for raw results")
    ap.add_argument("--model", default=os.getenv("DISEVAL_MODEL", DEFAULT_MODEL))
    ap.add_argument("--limit", type=int, default=None, help="only run the first N sentences")
    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Put it in a .env file (see .env.example).")

    client = anthropic.Anthropic(max_retries=5)
    examples = load_split(args.data, args.limit)
    done = _done_ids(args.out)
    todo = [e for e in examples if e.id not in done]
    print(f"{len(examples)} sentences, {len(done)} already done, {len(todo)} to run ({args.model})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as out:
        for n, ex in enumerate(todo, 1):
            row = {"id": ex.id, "text": ex.text, "gold": ex.gold, "pred": [], "error": None,
                   "model": args.model, "prompt_version": PROMPT_VERSION}
            try:
                row["pred"] = extract_one(client, args.model, ex.text)
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                    anthropic.NotFoundError) as e:
                sys.exit(f"Stopping: {type(e).__name__}: {e}")  # bad key or model name
            except Exception as e:  # keep going, record the failure
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
