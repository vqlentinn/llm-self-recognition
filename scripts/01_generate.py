"""Phase 1 — Generate responses for each model on each prompt.

Saves to data/generated/{model_slug}.csv (resumable: skips already done).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT / "data" / "prompts.json"
OUT_DIR = ROOT / "data" / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.3-70b-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "microsoft/phi-4",
    "qwen/qwen2.5-7b-instruct",
    "qwen/qwen2.5-72b-instruct",
]

TEMPERATURE = 0.7
MAX_TOKENS = 400
SLEEP_BETWEEN = 1.6  # ~40 req/min


def model_slug(model: str) -> str:
    return model.replace("/", "__")


def generate_response(client: OpenAI, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def main() -> None:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("Set NVIDIA_API_KEY in .env")

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )

    prompts = json.loads(PROMPTS_PATH.read_text())
    print(f"{len(prompts)} prompts, {len(MODELS)} models → {len(prompts)*len(MODELS)} calls total")

    for model in MODELS:
        slug = model_slug(model)
        out_path = OUT_DIR / f"{slug}.csv"

        # Load existing to resume
        if out_path.exists():
            existing = pd.read_csv(out_path)
            done_ids = set(existing["prompt_id"].tolist())
        else:
            existing = pd.DataFrame()
            done_ids = set()

        rows = existing.to_dict("records") if not existing.empty else []
        todo = [p for p in prompts if p["id"] not in done_ids]

        if not todo:
            print(f"[skip] {model} — already done")
            continue

        print(f"\n[{model}] {len(todo)} prompts to generate...")
        for prompt in tqdm(todo, desc=slug[:30]):
            try:
                response = generate_response(client, model, prompt["text"])
                rows.append({
                    "prompt_id": prompt["id"],
                    "category": prompt["category"],
                    "prompt": prompt["text"],
                    "model": model,
                    "response": response,
                })
                pd.DataFrame(rows).to_csv(out_path, index=False)
            except Exception as e:
                print(f"  [error] prompt {prompt['id']}: {e}")
            time.sleep(SLEEP_BETWEEN)

        print(f"  → saved {out_path}")

    print("\nDone. All responses in data/generated/")


if __name__ == "__main__":
    main()
