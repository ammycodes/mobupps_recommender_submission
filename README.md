# MobUpps Senior Data Scientist — Recommendation Modeling

This repository contains my solution for the MobUpps recommendation modeling take-home exercise.

The core deliverable is an **Item-Item Collaborative Filtering** model designed for mobile app discovery from implicit interaction logs. It incorporates event weighting, log-frequency dampening, exponential temporal decay, and shrinkage to prevent spurious correlations on small catalogs.

Detailed methodology, evaluation critique, hyperparameter analysis, and online A/B test design are documented in [**`ANALYSIS.md`**](ANALYSIS.md).

---

## Quickstart

### 1. Environment Setup

Python 3.10+ is recommended. Create a virtual environment and install dependencies:

```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Data Placement

Place the provided dataset files inside a `data/` directory at the project root:

```text
data/
  users.csv
  apps.csv
  interactions.csv
```

*(Note: Data files are excluded from Git via `.gitignore`.)*

### 3. Run Benchmark & Evaluation

Run the evaluation script from the repository root:

```bash
python scripts/evaluate.py --data-dir data --output evaluation_results.json
```

This will:
1. Run data quality audits on raw inputs.
2. Split interactions per user (leave-last-20%-out for users with $\ge 5$ interactions).
3. Evaluate the **Popularity Baseline**, **Item-Item CF**, **SVD Matrix Factorization**, and a blended **Ensemble**.
4. Log execution details to `logs/evaluate.log`.
5. Save fitted models into `models/*.pkl`.
6. Output metrics to `evaluation_results.json` and print the summary table.

---

## Offline Benchmark Results

Evaluated on 8,776 test users with $k=10$:

| Model | Recall@10 | NDCG@10 | Relative NDCG Lift | Fit Time |
| :--- | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0790 | 0.0876 | — | 0.08s |
| **Item-Item CF (Selected)** | **0.1122** | **0.1172** | **+33.8%** | 1.27s |
| **SVD Matrix Factorization** | 0.0791 | 0.0829 | -5.4% | 2.21s |
| **Ensemble (0.4 II + 0.6 SVD)** | 0.1028 | 0.1030 | +17.6% | 2.99s |

*Item-Item CF achieved the highest Recall@10 and NDCG@10. On a compact catalog of ~500 apps, direct item co-occurrence with shrinkage captures niche affinities accurately without the latent-factor bleeding observed in SVD.*

---

## Python API Usage

All models implement the required `fit(train_df)` and `recommend(user_id, n)` contract:

```python
import pandas as pd
from src.model import ItemItemRecommender

# Load training data
train_df = pd.read_csv("data/interactions.csv")

# Train and infer
model = ItemItemRecommender(shrinkage=10.0, popularity_blend=0.02)
model.fit(train_df)

# Recommend top 10 apps for a user
recommendations = model.recommend(user_id="u_000001", n=10)
print("Top 10 apps:", recommendations)

# Persistence: Save and load
model.save("models/item_item_cf.pkl")
loaded = ItemItemRecommender.load("models/item_item_cf.pkl")
```

- **Cold / Unseen Users:** Fall back automatically to popularity recommendations.
- **Seen Items:** Apps already present in the user's training history are masked out during recommendation.

---

## Warehouse SQL Queries

Production training feature extraction queries are provided in [`sql/training_features.sql`](sql/training_features.sql):
- User-app weighted event aggregation with event-type weights.
- Point-in-time app eligibility joins enforcing active campaign status.

---

## Project Structure

```text
mobupps_recommender_submission/
├── ANALYSIS.md                 # Core written analysis (Sections 5.1-5.5 & 6)
├── README.md                   # Setup, benchmarks, and API usage
├── requirements.txt            # Python dependencies (numpy, pandas, scipy)
├── .gitignore                  # Ignores data/, .venv/, models/*.pkl, logs/
├── data/                       # Local data directory (not committed)
├── logs/                       # Evaluation execution logs
├── models/                     # Serialized model artifacts (.pkl)
├── sql/
│   └── training_features.sql   # Production SQL queries
├── src/
│   ├── __init__.py
│   ├── data.py                 # Data loading, validation, and audit checks
│   └── model.py                # Recommender implementations (Item-Item, SVD, Ensemble)
└── scripts/
    └── evaluate.py             # Evaluation harness, benchmark & model saver
```
