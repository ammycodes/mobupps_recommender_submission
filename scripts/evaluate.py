from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_interactions, load_dimensions, audit_dimensions
from src.model import ItemItemRecommender, SVDRecommender, EnsembleRecommender


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    min_interactions: int = 5,
):
    train_parts, test_parts = [], []
    for _, group in df.groupby("user_id", sort=False):
        group = group.sort_values("event_time", kind="mergesort")
        if len(group) < min_interactions:
            train_parts.append(group)
            continue
        n_test = max(1, int(len(group) * test_fraction))
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])
    train = pd.concat(train_parts, ignore_index=True)
    test = (pd.concat(test_parts, ignore_index=True) if test_parts
            else pd.DataFrame(columns=df.columns))
    return train, test


def score(model, test_df: pd.DataFrame, k: int = 10) -> dict:
    actuals = test_df.groupby("user_id")["app_id"].apply(set).to_dict()
    recalls, ndcgs = [], []
    for user_id, actual in actuals.items():
        predicted = model.recommend(str(user_id), k)
        recalls.append(len(set(predicted[:k]) & actual) / len(actual) if actual else 0.0)
        dcg  = sum(1.0 / math.log2(i + 2)
                   for i, app in enumerate(predicted[:k]) if app in actual)
        ideal = min(len(actual), k)
        idcg  = sum(1.0 / math.log2(i + 2) for i in range(ideal))
        ndcgs.append(dcg / idcg if idcg else 0.0)
    return {
        "recall_at_10": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "ndcg_at_10":   float(sum(ndcgs)   / len(ndcgs))   if ndcgs   else 0.0,
        "users_evaluated": len(actuals),
    }


def _fit_and_score(name: str, model, train_df, test_df, k: int) -> dict:
    """Fit a model, evaluate it, log timing, and return results dict."""
    logger.info("--- Fitting %s ---", name)
    t0 = time.perf_counter()
    model.fit(train_df)
    fit_sec = time.perf_counter() - t0
    logger.info("%s fit in %.1fs", name, fit_sec)

    logger.info("--- Scoring %s ---", name)
    t0 = time.perf_counter()
    metrics = score(model, test_df, k=k)
    score_sec = time.perf_counter() - t0
    logger.info(
        "%s: Recall@%d=%.4f  NDCG@%d=%.4f  (score in %.1fs)",
        name, k, metrics["recall_at_10"], k, metrics["ndcg_at_10"], score_sec,
    )
    metrics["fit_seconds"]   = round(fit_sec, 2)
    metrics["score_seconds"] = round(score_sec, 2)
    return metrics


class PopularityBaseline:
    def fit(self, train_df: pd.DataFrame):
        self.apps = train_df["app_id"].value_counts().index.astype(str).tolist()
        return self

    def recommend(self, user_id: str, n: int) -> list[str]:
        return self.apps[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark recommender models.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output",   default="evaluation_results.json")
    parser.add_argument("--model-dir", default="models",
                        help="Directory to save fitted models as .pkl files")
    parser.add_argument("--log-file",  default="logs/evaluate.log")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--svd-factors", type=int, default=64)
    args = parser.parse_args()

    setup_logging(Path(args.log_file))
    logger.info("=" * 60)
    logger.info("MobUpps Recommender Evaluation  k=%d", args.k)
    logger.info("=" * 60)

    # --- Load data --------------------------------------------------------
    data_dir = Path(args.data_dir)
    logger.info("Loading interactions from %s", data_dir / "interactions.csv")
    interactions, dq_report = load_interactions(data_dir / "interactions.csv")
    users_df, apps_df = load_dimensions(data_dir)
    dq_report.update(audit_dimensions(interactions, users_df, apps_df))
    logger.info("Data quality: %s", dq_report)

    train, test = load_test_split(interactions)
    split_info = {
        "train_rows": len(train),
        "test_rows":  len(test),
        "test_users": int(test["user_id"].nunique()),
    }
    logger.info("Split: train=%d  test=%d  test_users=%d",
                split_info["train_rows"], split_info["test_rows"], split_info["test_users"])

    # --- Benchmark models -------------------------------------------------
    model_dir = Path(args.model_dir)
    k = args.k

    results: dict = {
        "data_quality": dq_report,
        "split": split_info,
        "caveat": (
            "Metrics use all held-out event types as binary relevance, including "
            "post-install events. They are offline ranking diagnostics, not a "
            "causal estimate of business impact."
        ),
    }

    # Popularity baseline (no save — trivial to reconstruct)
    baseline = PopularityBaseline()
    results["popularity_baseline"] = _fit_and_score(
        "PopularityBaseline", baseline, train, test, k
    )

    # Item-item CF
    ii_model = ItemItemRecommender()
    results["item_item_cf"] = _fit_and_score(
        "ItemItemCF", ii_model, train, test, k
    )
    ii_model.save(model_dir / "item_item_cf.pkl")

    # SVD matrix factorisation
    svd_model = SVDRecommender(n_factors=args.svd_factors)
    results["svd_mf"] = _fit_and_score(
        "SVD-MF", svd_model, train, test, k
    )
    svd_model.save(model_dir / "svd_mf.pkl")

    # Ensemble
    ensemble = EnsembleRecommender(alpha=0.4)
    results["ensemble"] = _fit_and_score(
        "Ensemble", ensemble, train, test, k
    )
    ensemble.save(model_dir / "ensemble.pkl")

    # --- Pick & log best model --------------------------------------------
    model_keys = ["item_item_cf", "svd_mf", "ensemble"]
    best_key = max(model_keys, key=lambda k_: results[k_]["ndcg_at_10"])
    results["best_model"] = best_key
    logger.info("Best model by NDCG@%d: %s (%.4f)", k, best_key, results[best_key]["ndcg_at_10"])

    # --- Print summary table ----------------------------------------------
    logger.info("\n%s", "=" * 60)
    logger.info("%-25s  %10s  %10s", "Model", f"Recall@{k}", f"NDCG@{k}")
    logger.info("-" * 50)
    for key in ["popularity_baseline", "item_item_cf", "svd_mf", "ensemble"]:
        r = results[key]
        logger.info("%-25s  %10.4f  %10.4f", key, r["recall_at_10"], r["ndcg_at_10"])
    logger.info("=" * 60)

    # --- Save results JSON ------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", output_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
