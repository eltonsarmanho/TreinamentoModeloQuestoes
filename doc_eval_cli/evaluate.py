#!/usr/bin/env python3
"""CLI that scores plain text/markdown documents on multiple dimensions via an OpenRouter-hosted LLM."""

import argparse
import json
import os
import sys
from dataclasses import dataclass

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"

DIMENSIONS = {
    "clarity": {
        "label": "Clarity",
        "instructions": "How clear and easy to follow is the writing in this document?",
        "criteria": [
            "Confusing; a reader would struggle to follow the structure or meaning",
            "Understandable but has rough passages, ambiguity, or awkward structure",
            "Clear and easy to follow throughout",
        ],
    },
    "correctness": {
        "label": "Correctness",
        "instructions": (
            "Based on general knowledge, how factually accurate and internally "
            "consistent are the claims made in this document?"
        ),
        "criteria": [
            "Contains clear factual errors or internal contradictions",
            "Mostly accurate but has a questionable or unverifiable claim",
            "Accurate and internally consistent throughout",
        ],
    },
}


@dataclass
class DocResult:
    path: str
    scores: dict


def read_documents(paths: list[str]) -> list[tuple[str, str]]:
    docs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            docs.append((path, f.read()))
    return docs


def build_schema(dimension_ids: list[str]) -> dict:
    properties = {}
    for dim_id in dimension_ids:
        criteria = DIMENSIONS[dim_id]["criteria"]
        properties[dim_id] = {
            "type": "object",
            "properties": {
                "level": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": len(criteria) - 1,
                    "description": "Index of the closest matching level below",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Self-assessed confidence in the level chosen",
                },
                "reasoning": {
                    "type": "string",
                    "description": "One short sentence justifying the level",
                },
            },
            "required": ["level", "confidence", "reasoning"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": properties,
        "required": dimension_ids,
        "additionalProperties": False,
    }


def build_prompt(dimension_ids: list[str]) -> str:
    lines = [
        "Score the given document on each listed dimension. For each dimension, "
        "pick the single closest level (0-indexed) from its list of levels.",
        "",
    ]
    for dim_id in dimension_ids:
        dim = DIMENSIONS[dim_id]
        lines.append(f'Dimension "{dim_id}": {dim["instructions"]}')
        for i, level in enumerate(dim["criteria"]):
            lines.append(f"  {i}: {level}")
        lines.append("")
    return "\n".join(lines)


def evaluate_document(api_key: str, model: str, text: str, dimension_ids: list[str]) -> dict:
    schema = build_schema(dimension_ids)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_prompt(dimension_ids)},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "document_scores", "schema": schema, "strict": True},
        },
    }
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def print_table(results: list[DocResult], dimension_ids: list[str]) -> None:
    headers = ["document"] + [DIMENSIONS[d]["label"] for d in dimension_ids]
    rows = []
    for result in results:
        row = [result.path]
        for dim_id in dimension_ids:
            answer = result.scores[dim_id]
            row.append(f"{answer['level']} (conf {answer['confidence']:.2f})")
        rows.append(row)

    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows)] if rows else [len(h) for h in headers]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate documents on multiple dimensions via an OpenRouter-hosted LLM."
    )
    parser.add_argument("paths", nargs="+", help="Paths to plain text or markdown files to evaluate")
    parser.add_argument(
        "--dimensions",
        nargs="+",
        choices=list(DIMENSIONS.keys()),
        default=list(DIMENSIONS.keys()),
        help="Which dimensions to score (default: all)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model id to use (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPEN_ROUTE_API_KEY")
    if not api_key:
        print("error: OPEN_ROUTE_API_KEY is not set", file=sys.stderr)
        return 1

    docs = read_documents(args.paths)

    results = []
    for path, text in docs:
        scores = evaluate_document(api_key, args.model, text, args.dimensions)
        results.append(DocResult(path=path, scores=scores))

    print_table(results, args.dimensions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
