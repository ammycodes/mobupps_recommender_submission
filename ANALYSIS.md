# Recommendation Modeling & Experimentation Analysis

## 5.1 Target and signal definition

In mobile ad-tech, we deal almost entirely with implicit feedback. Users don't give 5-star ratings; they view an ad, click, install, or make in-app purchases. Treating all these interactions equally would be a mistake—a view is often passive exposure, while an install or purchase indicates strong intent.

For the training signal, I constructed a composite interaction score using business-guided weights:
- `af_view`: 0.2
- `af_click` / `af_open`: 1.0
- `af_complete_registration`: 3.0
- `af_install`: 3.0
- `af_level_achieved`: 2.0
- `af_purchase`: 4.0

To keep power users or automated bot-like clicks from distorting the item similarities, I dampen raw totals using `log(1 + weighted_score)`. I also apply an exponential time decay with a 30-day half-life (`exp(-λ * days)`). App preferences decay quickly in mobile; an install 70 days ago shouldn't have the same predictive weight as an app opened yesterday.

**The unobserved pair problem:** We never observe true negative feedback. If a user hasn't interacted with an app, we don't know whether they disliked it, never saw it, or simply didn't have time. My model treats unobserved user-app pairs as zero. 

What does this assumption cost us?
1. **Popularity bias & exposure confounding:** Heavily displayed apps get more feedback simply by virtue of being shown, while niche apps look like "zeros" due to lack of exposure.
2. **Filtering out consumed items:** At serving time, I mask out apps the user already interacted with during training. For utility apps (like calculators or ride sharing), you rarely want to recommend what someone already installed. However, for recurring-use games or apps with frequent re-engagement campaigns, strictly excluding prior apps might artificially suppress high-converting recommendations.

---

## 5.2 Model choice and the one you rejected

### Chosen Model: Item-Item Collaborative Filtering with Shrinkage

Looking at the dataset dimensions:
- ~500 unique apps
- ~10,000 users
- ~490,000 interactions (~9.8% user-app density, but heavily skewed)

With only ~500 items, the item-item similarity matrix is tiny (500 × 500, under 2 MB in memory). Item-Item CF is practically made for this setup:
1. **Extreme efficiency:** Fitting takes ~1.2 seconds, and offline scoring takes under 2 seconds across 8,700+ users. In production, the 500×500 similarity matrix can be precomputed offline in minutes, pushed to a Redis cache, and queried in sub-millisecond time.
2. **Explainability:** In ad-tech, ad operations and account managers need to know *why* an app was recommended (e.g., "Recommended because you installed App X"). Item-item makes attribution straightforward.
3. **Handling sparse users:** For users with only 1 or 2 historical interactions, item-item immediately retrieves nearest-neighbor apps without needing dense user embeddings.

### Empirical Comparison

I also implemented and benchmarked a Truncated SVD matrix factorization (`k=64`) and a blended Ensemble on the exact same leave-last-20% split:

| Model | Recall@10 | NDCG@10 | Lift vs Baseline | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Popularity Baseline** | 0.0790 | 0.0876 | — | Simple frequency prior |
| **Item-Item CF (Chosen)** | **0.1122** | **0.1172** | **+33.8%** | Best ranking quality & explainability |
| **SVD Matrix Factorization** | 0.0791 | 0.0829 | -5.4% | Smooths out distinct niche clusters |
| **Ensemble (0.4 II + 0.6 SVD)** | 0.1028 | 0.1030 | +17.6% | Solid, but worse than pure Item-Item |

*Why did SVD underperform here?* SVD projects the user-app matrix into a lower-dimensional latent space. With only ~500 apps, many apps belong to distinct isolated clusters (e.g., casual puzzles, crypto trading, food delivery). SVD forces these distinct co-occurrence patterns into continuous factors, which causes "bleed" across unrelated categories and depresses precision on tail apps. Item-item similarity with shrinkage respects local graph structure much better.

### Deliberately Rejected: Deep Neural / Sequential Transformers (e.g., SASRec, BERT4Rec)

I intentionally did not build a sequence-based deep learning model. 
1. **Catalog size:** With only ~500 items, learning dense token embeddings with multi-head self-attention is massive overkill and prone to severe overfitting.
2. **Session length:** Most users in this dataset have relatively short histories. Sequence models need rich, granular session telemetry to learn transition dynamics.
3. **Operational overhead:** Serving a Transformer in real-time adds GPU dependencies, inference latency (tens of milliseconds vs. sub-millisecond lookup), and complex CI/CD pipelines for negligible or negative offline gain.

---

## 5.3 Critique the evaluation

The offline harness uses a per-user leave-last-20%-out split and evaluates Recall@10 and NDCG@10. While standard in academic recsys literature, this setup has serious blind spots from a business perspective:

1. **The split allows temporal leakage across users:**
   Because the split is per-user rather than a fixed global timestamp cutoff, user A's training set can include events that occurred *after* user B's test set. In production, you never train on future data to predict the past. A true backtest must use a single point-in-time cutoff (e.g., train on everything before day 70, evaluate on days 70–90).
2. **Equal weight to all test events:**
   The evaluation treats any held-out event as a positive binary hit. A user passively viewing an app banner counts as much as an install or a $50 in-app purchase. In reality, MobUpps makes money on installs and downstream retention, not casual views. Optimizing Recall@10 on this harness can encourage the model to recommend high-click/high-view clickbait apps that fail to convert downstream.
3. **No negative feedback or cannibalization:**
   The offline test cannot tell us if the user would have found the app anyway (organic lift vs. paid cannibalization). It also ignores app availability (whether campaigns are currently active or budget-capped).

**Would I trust these numbers to decide a production rollout?**
No. I trust them as a directional filter to discard degenerate models (e.g., proving our personalization comfortably beats random or pure popularity). But I would never ship straight to 100% production based on offline NDCG alone. The gap between offline proxy metrics and real ad revenue / ROAS is too large.

---

## 5.4 Hyperparameters: predict, don't list

For the Item-Item model, the two hyperparameters that matter most are:

### 1. Similarity Shrinkage (`shrinkage`, set to 10.0)
This parameter damps cosine similarity when the number of co-occurring users between two apps is small:
$$s'(i, j) = \frac{C(i, j)}{C(i, j) + \text{shrinkage}} \cdot \text{cosine}(i, j)$$

- **What happens as it increases:** It aggressively penalizes similarities supported by few shared users, pushing scores toward zero unless supported by thick interaction volume.
- **Offline metric trajectory:** In the evaluation harness, as shrinkage increases from 0 to ~10–20, NDCG@10 goes up because spurious coincidences (e.g., two users happened to view two obscure apps) are filtered out. Beyond that, NDCG plateaus and starts dropping as niche similarities get wiped out.
- **Real-world quality & failure modes:**
  - *At 0 (no shrinkage):* The model recommends obscure, irrelevant apps based on accidental 1-user overlaps. Quality feels erratic and untrustworthy.
  - *At $\infty$ (extreme shrinkage):* All item-item similarities collapse to zero. The model degrades entirely into a global popularity ranker, losing all personalization.

### 2. Popularity Blend (`popularity_blend`, set to 0.02)
This weight blends the global normalized popularity score into the final item score for sparse or cold profiles.

- **What happens as it increases:** It shifts recommendations from personalized item-item affinities toward the overall top-performing apps across the network.
- **Offline metric trajectory:** Mild blending (0.01–0.05) helps marginally on Recall@10 by filling the bottom of the top-10 list with high-base-rate apps when a user has only 1 interaction. As it increases further (0.2+), offline NDCG drops significantly because personalized hits are pushed out of the top ranks.
- **Real-world quality & failure modes:**
  - *At 0:* Cold or sparse users get truncated or empty recommendation lists if their past apps have zero similarity to anything else.
  - *At 1.0:* Every single user gets the exact same top-10 popular apps (Instagram, TikTok, Subway Surfers). Niche user segments receive zero relevant personalization, eroding click-through rates.

---

## 5.5 From offline to online

Here is how I would design the online A/B test to validate shipping the model:

### 1. Hypothesis
Serving personalized recommendations via Item-Item CF will increase the downstream install rate and user engagement compared to the current production baseline (popularity/heuristic ranking), without degrading latency or ad inventory fill rate.

### 2. Primary Metric
**Attributed Install Conversion Rate (CVR):** 
$$\text{CVR} = \frac{\text{Unique Users with an Install Event within 24h}}{\text{Unique Users Served Recommendation Impressions}}$$
*Why this metric:* In mobile advertising, clicks are cheap and easily gamed by sensational creative. Installs require active user commitment and reflect genuine intent. 

### 3. Randomization Unit
**User ID (hash-bucketed with a salt):**
Randomizing at the request or impression level would cause severe cross-contamination (the same user seeing different recommendation strategies across app opens, confusing their experience). We hash `user_id` into 100 buckets: 50% Control (Current Production), 50% Treatment (Item-Item CF).

### 4. Experiment Sizing
- Baseline CVR: ~3.0% (estimated from historical traffic).
- Minimum Detectable Effect (MDE): 5% relative lift (from 3.0% to 3.15%).
- Statistical power: 80%, $\alpha = 0.05$ (two-sided).
- Using a standard two-proportion sample size formula, this requires approximately **115,000 eligible users per variant**.
- Run duration: Minimum **14 full days** to capture two complete weekly cycles (weekday vs. weekend behavior varies drastically in mobile gaming/apps) and allow full attribution windows to close.

### 5. Guardrail Metrics
1. **P99 Inference Latency:** Must remain under 25ms. If recommendations cause timeouts or UI stutter, user drop-off will kill conversions regardless of model accuracy.
2. **Campaign Diversity / Catalog Coverage:** The percentage of active advertiser budgets receiving impressions. If the recommender over-concentrates on 5 popular apps, advertiser churn and unspent budgets will harm network revenue.

### 6. Biggest Validity Threat
**Attribution Window & Delayed Conversions (Feedback Loop Delay).**
Ad-tech attribution windows often span 24 hours to 7 days (via SKAdNetwork on iOS or Google Play Install Referrer). If treatment users install apps several days after the impression, evaluating early or peeking at daily metrics leads to false negatives or premature stopping. Furthermore, inventory bidding dynamics (eCPM floors) can shift during the test, meaning treatment might win different impression slots than control if not strictly isolated.

---

## 6. Data-Quality Findings

During data audit and preparation, I identified the following issues:

1. **493 invalid / unparseable timestamps:** 
   - *How found:* Strict ISO-8601 parsing (`pd.to_datetime(errors='coerce')`).
   - *Action:* **Flagged and excluded** from train/test split. Ordering is unknowable for corrupted timestamps, so treating them as temporal points would contaminate the split.
2. **1,887 exact duplicate rows:**
   - *How found:* `df.duplicated()` check across all interaction columns.
   - *Action:* **Flagged but preserved** in initial counts; deduplicated per minute window during feature engineering. A user opening or clicking an app twice in 5 minutes can be a genuine repeated engagement rather than a DB glitch.
3. **245 interactions with orphan user IDs and 245 with orphan app IDs:**
   - *How found:* Anti-joins against `users.csv` and `apps.csv`.
   - *Action:* **Flagged and allowed in CF matrix**, but flagged for data engineering. Collaborative filtering only needs interaction IDs, but downstream profile-based targeting would drop these rows.
4. **App Status Distribution (Active / Paused / Blocked):**
   - *How found:* Groupby on `apps['status']`.
   - *Action:* **Documented.** The dataset contains current status, but not the status history at the time each interaction occurred. In production, an app eligibility filter must run at serving time to prevent bidding on paused campaigns.
