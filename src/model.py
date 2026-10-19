from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class ItemItemRecommender:
    """
    Weighted item-item collaborative filtering — enhanced.

    Improvements over the original baseline:
      1. Temporal decay   – recent events are weighted higher via exp(-λ·days).
      2. IDF weighting    – apps seen by many users are down-weighted
                            (BM25-style), reducing popularity bias in similarity.
      3. Asymmetric sim   – optionally uses P(B | seen A) instead of symmetric
                            cosine; better models directional install likelihood.
      4. Adaptive pop     – sparse users get a stronger popularity blend;
                            active users rely more on personalisation.
    """

    EVENT_WEIGHTS: dict[str, float] = {
        "af_view": 0.2,
        "af_click": 1.0,
        "af_open": 1.0,
        "af_install": 3.0,
        "af_purchase": 4.0,
        "af_complete_registration": 3.0,
        "af_level_achieved": 2.0,
    }

    def __init__(
        self,
        shrinkage: float = 10.0,
        popularity_blend: float = 0.05,
        # Temporal decay: half-life in days (None = disabled)
        decay_halflife_days: float | None = 30.0,
        # IDF smoothing factor (0 = disabled, 1 = standard)
        idf_alpha: float = 0.5,
        # Use asymmetric P(B|A) similarity instead of cosine
        asymmetric: bool = False,
        # Sparse user threshold: n_distinct_apps below which pop blend is boosted
        sparse_threshold: int = 5,
        sparse_blend_boost: float = 0.20,
    ):
        if shrinkage < 0:
            raise ValueError("shrinkage must be non-negative")
        if not 0 <= popularity_blend <= 1:
            raise ValueError("popularity_blend must be between 0 and 1")
        self.shrinkage = float(shrinkage)
        self.popularity_blend = float(popularity_blend)
        self.decay_halflife_days = decay_halflife_days
        self.idf_alpha = float(idf_alpha)
        self.asymmetric = asymmetric
        self.sparse_threshold = sparse_threshold
        self.sparse_blend_boost = float(sparse_blend_boost)
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_decay(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multiply _weight by exp(-λ·days_since_event)."""
        if self.decay_halflife_days is None or "event_time" not in df.columns:
            return df
        ref = df["event_time"].max()
        days = (ref - df["event_time"]).dt.total_seconds() / 86_400.0
        lam = np.log(2) / self.decay_halflife_days
        df = df.copy()
        df["_weight"] = df["_weight"] * np.exp(-lam * days.clip(lower=0))
        return df

    def _build_idf(self, matrix: np.ndarray, n_users: int) -> np.ndarray:
        """IDF per app: log((n_users+1)/(doc_freq+1)) + 1, blended with alpha."""
        if self.idf_alpha == 0:
            return np.ones(matrix.shape[1], dtype=np.float32)
        doc_freq = (matrix > 0).sum(axis=0).astype(np.float64)
        idf = np.log((n_users + 1) / (doc_freq + 1)) + 1.0
        # alpha blending: 0 = no IDF, 1 = full IDF
        idf = self.idf_alpha * idf + (1.0 - self.idf_alpha)
        return idf.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the fitted model to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("ItemItemRecommender saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ItemItemRecommender":
        """Load a pickled ItemItemRecommender from *path*."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("ItemItemRecommender loaded from %s", path)
        return obj

    def fit(self, train_df: pd.DataFrame) -> "ItemItemRecommender":
        required = {"user_id", "app_id"}
        if not required.issubset(train_df.columns):
            raise ValueError(f"train_df must contain {sorted(required)}")

        logger.info("ItemItemRecommender.fit: %d rows", len(train_df))
        df = train_df.copy()
        df["user_id"] = df["user_id"].astype(str)
        df["app_id"] = df["app_id"].astype(str)

        # --- event weights ------------------------------------------------
        if "event_name" in df.columns:
            df["_weight"] = df["event_name"].map(self.EVENT_WEIGHTS).fillna(1.0)
        else:
            df["_weight"] = 1.0

        # --- temporal decay -----------------------------------------------
        if "event_time" in df.columns:
            df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
            df = df.dropna(subset=["event_time"])
        df = self._apply_decay(df)

        # --- aggregate & dampen -------------------------------------------
        aggregated = df.groupby(["user_id", "app_id"], sort=False)["_weight"].sum()
        users = aggregated.index.get_level_values("user_id").unique().tolist()
        apps  = aggregated.index.get_level_values("app_id").unique().tolist()
        user_index = {u: i for i, u in enumerate(users)}
        app_index  = {a: i for i, a in enumerate(apps)}

        rows = aggregated.index.get_level_values("user_id").map(user_index).to_numpy()
        cols = aggregated.index.get_level_values("app_id").map(app_index).to_numpy()
        vals = np.log1p(aggregated.to_numpy(dtype=np.float64))

        # Dense matrix — small catalogue (~500 apps) keeps this cheap.
        n_users_total = len(users)
        matrix = np.zeros((n_users_total, len(apps)), dtype=np.float32)
        matrix[rows, cols] = vals.astype(np.float32)

        # --- IDF weighting ------------------------------------------------
        idf = self._build_idf(matrix, n_users_total)   # shape: (n_apps,)
        matrix_idf = matrix * idf[np.newaxis, :]        # broadcast over users

        # --- similarity ---------------------------------------------------
        if self.asymmetric:
            # Directed: sim[A, B] = co_support(A,B) / support(A)
            # Captures P(user interacts with B | already interacted with A)
            binary = (matrix_idf > 0).astype(np.float32)
            co = binary.T @ binary                          # (apps x apps)
            support_a = binary.sum(axis=0)                  # (apps,)
            support_a[support_a == 0] = 1.0
            similarity = co / support_a[:, np.newaxis]      # row-normalise
            np.fill_diagonal(similarity, 0.0)
        else:
            # Standard cosine on IDF-weighted matrix
            norms = np.sqrt(np.sum(matrix_idf * matrix_idf, axis=0))
            norms[norms == 0] = 1.0
            co = matrix_idf.T @ matrix_idf
            similarity = co / np.outer(norms, norms)
            np.fill_diagonal(similarity, 0.0)

        # --- shrinkage ----------------------------------------------------
        if self.shrinkage:
            binary = (matrix > 0).astype(np.float32)
            support = binary.T @ binary
            similarity *= support / (support + self.shrinkage)

        # --- per-user distinct-app count (for adaptive blend) ------------
        user_event_counts = (matrix > 0).sum(axis=1)   # shape: (n_users,)

        # --- popularity prior --------------------------------------------
        counts = df.groupby("app_id").size()
        popularity = np.array([counts.get(a, 0) for a in apps], dtype=np.float64)
        if popularity.max() > 0:
            popularity /= popularity.max()

        # --- store state -------------------------------------------------
        self.users_ = users
        self.apps_  = apps
        self.user_index_         = user_index
        self.app_index_          = app_index
        self.user_app_matrix_    = matrix           # raw log1p (for seen-mask)
        self.user_app_matrix_idf = matrix_idf       # IDF-weighted (for scoring)
        self.similarity_         = similarity.astype(np.float32)
        self.popularity_         = popularity
        self.user_event_counts_  = user_event_counts
        self._fitted = True
        logger.info(
            "ItemItemRecommender fitted: %d users, %d apps",
            len(users), len(apps),
        )
        return self

    def recommend(self, user_id: str, n: int) -> list[str]:
        if not self._fitted:
            raise RuntimeError("Call fit(train_df) before recommend().")
        if n <= 0:
            return []
        user_id = str(user_id)

        # --- cold-start: unknown user → pure popularity ------------------
        if user_id not in self.user_index_:
            order = np.argsort(-self.popularity_, kind="stable")
            return [self.apps_[i] for i in order[:n]]

        uid         = self.user_index_[user_id]
        history     = self.user_app_matrix_[uid]        # raw weights (seen-mask)
        history_idf = self.user_app_matrix_idf[uid]     # IDF-weighted (scoring)
        seen        = history > 0

        # --- personalised scores -----------------------------------------
        scores = history_idf @ self.similarity_
        scores = scores.astype(np.float64)

        # --- adaptive popularity blend -----------------------------------
        n_distinct = int(self.user_event_counts_[uid])
        if n_distinct < self.sparse_threshold:
            blend = self.popularity_blend + self.sparse_blend_boost
        else:
            blend = self.popularity_blend
        scores += blend * self.popularity_

        # --- mask already-seen apps --------------------------------------
        scores[seen] = -np.inf
        order = np.argsort(-scores, kind="stable")
        recs = [self.apps_[i] for i in order[:n] if np.isfinite(scores[i])][:n]
        logger.debug("ItemItemRecommender.recommend(%s, %d) -> %d results", user_id, n, len(recs))
        return recs


# ---------------------------------------------------------------------------
# SVD-based Matrix Factorisation Recommender
# ---------------------------------------------------------------------------

class SVDRecommender:
    """
    Truncated SVD (latent-factor) collaborative filtering for implicit feedback.

    Builds the same weighted, log1p-dampened, time-decayed user-app matrix as
    ItemItemRecommender, then factorises it with truncated SVD:

        M \u2248 U \u00b7 \u03a3 \u00b7 V^T

    User scores are computed as  (U \u00b7 diag(\u03a3)) @ V^T, which gives dense
    personalised rankings without needing to materialise an items x items
    similarity matrix.  This generalises better for users with thin history
    because latent factors pool evidence across the whole catalogue.

    Cold users fall back to the popularity prior.
    """

    EVENT_WEIGHTS: dict[str, float] = {
        "af_view": 0.2,
        "af_click": 1.0,
        "af_open": 1.0,
        "af_install": 3.0,
        "af_purchase": 4.0,
        "af_complete_registration": 3.0,
        "af_level_achieved": 2.0,
    }

    def __init__(
        self,
        n_factors: int = 64,
        decay_halflife_days: float | None = 30.0,
        popularity_blend: float = 0.05,
        sparse_threshold: int = 5,
        sparse_blend_boost: float = 0.20,
    ):
        self.n_factors = n_factors
        self.decay_halflife_days = decay_halflife_days
        self.popularity_blend = popularity_blend
        self.sparse_threshold = sparse_threshold
        self.sparse_blend_boost = sparse_blend_boost
        self._fitted = False

    def _apply_decay(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.decay_halflife_days is None or "event_time" not in df.columns:
            return df
        ref = df["event_time"].max()
        days = (ref - df["event_time"]).dt.total_seconds() / 86_400.0
        lam = np.log(2) / self.decay_halflife_days
        df = df.copy()
        df["_weight"] = df["_weight"] * np.exp(-lam * days.clip(lower=0))
        return df

    def fit(self, train_df: pd.DataFrame) -> "SVDRecommender":
        logger.info("SVDRecommender.fit: %d rows, n_factors=%d", len(train_df), self.n_factors)
        df = train_df.copy()
        df["user_id"] = df["user_id"].astype(str)
        df["app_id"] = df["app_id"].astype(str)

        if "event_name" in df.columns:
            df["_weight"] = df["event_name"].map(self.EVENT_WEIGHTS).fillna(1.0)
        else:
            df["_weight"] = 1.0

        if "event_time" in df.columns:
            df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
            df = df.dropna(subset=["event_time"])
        df = self._apply_decay(df)

        aggregated = df.groupby(["user_id", "app_id"], sort=False)["_weight"].sum()
        users = aggregated.index.get_level_values("user_id").unique().tolist()
        apps  = aggregated.index.get_level_values("app_id").unique().tolist()
        user_index = {u: i for i, u in enumerate(users)}
        app_index  = {a: i for i, a in enumerate(apps)}

        rows = aggregated.index.get_level_values("user_id").map(user_index).to_numpy()
        cols = aggregated.index.get_level_values("app_id").map(app_index).to_numpy()
        vals = np.log1p(aggregated.to_numpy(dtype=np.float64)).astype(np.float32)

        # Build sparse matrix and factorise
        sparse_m = csr_matrix((vals, (rows, cols)), shape=(len(users), len(apps)))
        k = min(self.n_factors, len(users) - 1, len(apps) - 1)
        U, sigma, Vt = svds(sparse_m, k=k)
        # svds returns in ascending singular-value order; reverse
        idx = np.argsort(-sigma)
        U, sigma, Vt = U[:, idx], sigma[idx], Vt[idx, :]

        # User latent vectors scaled by singular values
        self.user_factors_  = (U * sigma[np.newaxis, :]).astype(np.float32)  # (users, k)
        self.item_factors_  = Vt.T.astype(np.float32)                         # (apps,  k)

        # Dense original matrix for seen-mask and sparse-user detection
        dense = np.zeros((len(users), len(apps)), dtype=np.float32)
        dense[rows, cols] = vals
        self.user_app_matrix_  = dense
        self.user_event_counts_ = (dense > 0).sum(axis=1)  # (n_users,)

        counts = df.groupby("app_id").size()
        popularity = np.array([counts.get(a, 0) for a in apps], dtype=np.float64)
        if popularity.max() > 0:
            popularity /= popularity.max()

        self.users_      = users
        self.apps_       = apps
        self.user_index_ = user_index
        self.app_index_  = app_index
        self.popularity_ = popularity
        self._fitted = True
        logger.info("SVDRecommender fitted: %d users, %d apps, k=%d", len(users), len(apps), k)
        return self

    def recommend(self, user_id: str, n: int) -> list[str]:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        if n <= 0:
            return []
        user_id = str(user_id)

        if user_id not in self.user_index_:
            order = np.argsort(-self.popularity_, kind="stable")
            return [self.apps_[i] for i in order[:n]]

        uid    = self.user_index_[user_id]
        scores = self.user_factors_[uid] @ self.item_factors_.T  # (n_apps,)
        scores = scores.astype(np.float64)

        n_distinct = int(self.user_event_counts_[uid])
        blend = (self.popularity_blend + self.sparse_blend_boost
                 if n_distinct < self.sparse_threshold else self.popularity_blend)
        scores += blend * self.popularity_

        seen = self.user_app_matrix_[uid] > 0
        scores[seen] = -np.inf

        order = np.argsort(-scores, kind="stable")
        recs = [self.apps_[i] for i in order[:n] if np.isfinite(scores[i])][:n]
        logger.debug("SVDRecommender.recommend(%s, %d) -> %d results", user_id, n, len(recs))
        return recs

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("SVDRecommender saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SVDRecommender":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("SVDRecommender loaded from %s", path)
        return obj


# ---------------------------------------------------------------------------
# Ensemble: blends ItemItem + SVD scores
# ---------------------------------------------------------------------------

class EnsembleRecommender:
    """
    Linear blend of ItemItemRecommender and SVDRecommender scores.

    Both models score all apps for a user; the final ranking is:
        score = alpha * item_item_score + (1 - alpha) * svd_score

    Both score vectors are min-max normalised before blending so they are on
    the same scale.  Apps already seen in training are masked out.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        item_item_kwargs: dict | None = None,
        svd_kwargs: dict | None = None,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.ii  = ItemItemRecommender(**(item_item_kwargs or {}))
        self.svd = SVDRecommender(**(svd_kwargs or {}))
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> "EnsembleRecommender":
        logger.info("EnsembleRecommender.fit: alpha=%.2f", self.alpha)
        self.ii.fit(train_df)
        self.svd.fit(train_df)
        # Both models share the same app catalogue after fitting on same data
        self._fitted = True
        logger.info("EnsembleRecommender fitted")
        return self

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        """Normalise finite values to [0, 1]; leave -inf untouched."""
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return arr
        lo, hi = finite.min(), finite.max()
        out = arr.copy()
        if hi > lo:
            out[np.isfinite(arr)] = (finite - lo) / (hi - lo)
        else:
            out[np.isfinite(arr)] = 0.5
        return out

    def recommend(self, user_id: str, n: int) -> list[str]:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        # Fetch raw scores from each model internals
        uid_str = str(user_id)

        def _raw_scores(model: ItemItemRecommender | SVDRecommender) -> tuple[np.ndarray, list[str], np.ndarray]:
            if uid_str not in model.user_index_:
                scores = model.popularity_.copy().astype(np.float64)
                seen = np.zeros(len(model.apps_), dtype=bool)
            else:
                uid = model.user_index_[uid_str]
                if isinstance(model, SVDRecommender):
                    scores = (model.user_factors_[uid] @ model.item_factors_.T).astype(np.float64)
                else:
                    scores = (model.user_app_matrix_idf[uid] @ model.similarity_).astype(np.float64)
                scores += model.popularity_blend * model.popularity_
                seen = model.user_app_matrix_[uid] > 0
            return scores, model.apps_, seen

        ii_scores,  ii_apps,  ii_seen  = _raw_scores(self.ii)
        svd_scores, svd_apps, svd_seen = _raw_scores(self.svd)

        # Both models are trained on the same data → same app list, same order
        blended = self.alpha * self._minmax(ii_scores) + (1 - self.alpha) * self._minmax(svd_scores)
        seen    = ii_seen | svd_seen
        blended[seen] = -np.inf

        order = np.argsort(-blended, kind="stable")
        recs = [ii_apps[i] for i in order[:n] if np.isfinite(blended[i])][:n]
        logger.debug("EnsembleRecommender.recommend(%s, %d) -> %d results", uid_str, n, len(recs))
        return recs

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("EnsembleRecommender saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleRecommender":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("EnsembleRecommender loaded from %s", path)
        return obj
