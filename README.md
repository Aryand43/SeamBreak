# SeamBreak

Minimal, scalable **LangDB API pipeline** with one invariant interface.

## Setup

1) Install deps:

```bash
python -m pip install -r requirements.txt
```

2) Create/update `.env`:

- `LANGDB_API_KEY=...`
- `LANGDB_PROJECT_ID=...`

## Run

```bash
python test_pipeline.py
```

## Dataset-grounded deterministic attack (TruthfulQA)

- Dataset path is fixed at `Data/TruthfulQA.csv`
- Loader: `truthfulqa_loader.py`
- Attack (pure message transform): `attacks/self_contradiction.py`
- `test_pipeline.py` runs baseline vs attacked for each dataset row across all configured models (logs output text only)
