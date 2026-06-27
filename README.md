# Do LLMs Recognize Their Own Outputs?

Empirical study of LLM self-attribution: can a model identify which of two responses to the same prompt it generated itself?

## Research Question

When presented with two responses to the same question — one its own, one from another model — can an LLM correctly identify which is its?
We test this across 10 model families and 50 prompts, measuring accuracy against a 50% random baseline.

## Preliminary Results (Gemma-3 family, Paul Favier)

| Model | Accuracy | vs 50% baseline |
|---|---:|---|
| gemma-3-4b | 48.4% | below chance |
| gemma-3-12b | 49.9% | ≈ chance |
| gemma-3-27b | 52.1% | slightly above |

Observation: LLMs do not reliably recognize their own outputs — even large models perform near chance level.

## Setup

```bash
git clone https://github.com/VOTRE_USER/llm-self-recognition
cd llm-self-recognition
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your NVIDIA API key
```

Get a free API key at [build.nvidia.com](https://build.nvidia.com) (no credit card required).

## Pipeline

```bash
# 1. Generate responses for each model × prompt
python scripts/01_generate.py

# 2. Ask each model to identify its own response
python scripts/02_attribute.py

# 3. Compute accuracy, Wilson CIs, figures
python scripts/03_analyze.py
```

Results are saved to `data/results/`. Figures go to `paper/figures/`.

## Models tested

- Llama 3.1 8B / 3.3 70B
- Mistral 7B / Mixtral 8x7B
- Gemma 3 4B / 12B / 27B
- Phi-4
- Qwen 2.5 7B / 72B

## Project structure

```
data/
  prompts.json          50 prompts across 5 categories
  generated/            model responses (gitignored)
  results/              accuracy scores, CSVs
scripts/
  01_generate.py        Phase 1: generate responses
  02_attribute.py       Phase 2: self-attribution experiment
  03_analyze.py         Phase 3: stats + figures
paper/
  figures/              generated plots
  main.tex              paper (WIP)
```

## Authors

- Paul Favier
- Constantin Beraud
- Valentin Proux

## Status

Work in progress — targeting arXiv submission.
