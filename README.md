# Heavy Equipment Selling Price Prediction

Predicting resale prices for heavy equipment (Kaggle competition `26ds1000057`), scored by RMSLE, with a viva examination requiring every modeling decision to be verbally defensible.

## Result

- **Best leaderboard score: 0.18998 RMSLE**
- Viva eligibility threshold: < 0.20 ✅
- Score progression: 0.19338 → 0.19233 → 0.19220 → 0.19107 → 0.18998
- Leaderboard Rank: 231 out of 2593 students

## Constraints Shaping This Project

This isn't a "throw the strongest tool at it" project — every choice had to survive two filters:

1. **CPU-only submission.** The graded `FinalNotebook` runs on Kaggle's CPU kernel; GPU is only used for local iteration during development.
2. **Full verbal defensibility.** A viva panel can ask "why" about any line of code. This ruled out Optuna, out-of-fold stacking, and regex-heavy feature extraction in favor of simpler, explainable alternatives- even when a more automated approach might have scored marginally better.
3. **Reusable functions, not copy-pasted code.** The grading rubric requires all feature engineering and preprocessing to be implemented as functions or pipelines, applied identically to train and test, to avoid leakage and redundancy.

## Approach

**Models**: A weighted blend of **CatBoost (0.60)**, **LightGBM (0.35)**, and **XGBoost (0.05)**, each tuned via `RandomizedSearchCV` on a subsampled search phase, then refit on the full training set using its winning hyperparameters.

**Baselines for comparison**: Linear Regression, KNN, AdaBoost, Random Forest- used to validate that a boosted-tree blend was justified (Linear Regression scored notably worse, confirming non-linear structure in the data) and to demonstrate a complete model-selection process.

**Key feature**: A causal, leak-safe rolling average of same-product historical prices is the dominant signal by a wide margin. Two window lengths (180-day and 365-day) plus their ratio are included; the 365-day window ranks #1 by gain across all three blend models, ahead of the original 180-day version.

## Repository / Notebook Structure

The notebook (`26ds1000057-notebook-2026t2.ipynb`) is organized as:

1. Data loading & initial exploration
2. Cleaning strategy (duplicates, impossible values, outliers)
3. Missing value handling
4. Feature engineering (all via reusable functions — see below)
5. Preprocessing pipeline (encoding, categorical handling)
6. Baseline model comparison (7 models)
7. Hyperparameter tuning (CatBoost, LightGBM, XGBoost)
8. Final blend weight search
9. Full-data refit & submission
10. Conclusion (feature importance, challenges, final score)

## Feature Engineering (Reusable Functions)

All feature engineering is implemented as functions applied identically to `df` and `test`, per the rubric:

| Function | Purpose |
|---|---|
| `add_hours_features` | Machine age, operational hours transforms, zero-hours flag |
| `fill_operational_hours` | Train-derived median imputation for operational hours |
| `add_spec_text_parsing_features` | Descriptor length, numeric prefix extraction |
| `add_categorical_grouping_features` | Seasonal, utilization, and interaction categorical features |
| `build_rolling_lookup` / `add_rolling_avg` | Leak-safe causal rolling average (generalized for any grouping column and window length) |
| `build_asset_lookup` / `apply_asset_prior` | Prior price/count per physical asset |
| `build_baseclass_lookup` / `apply_baseclass_prior` | Prior mean price per equipment base class |

## Key Bugs Found and Fixed

- **XGBoost categorical re-indexing bug**: `enable_categorical=True` produced inconsistent category codes across train/val/test splits, distorting val RMSLE to ~0.347 (vs. an expected ~0.206). Fixed with a unified `pd.CategoricalDtype` built from the union of all three splits' values per column.
- **CatBoost final-refit bug**: the full-data refit model was using hardcoded hyperparameters instead of the actual `RandomizedSearchCV` results. Fixed to pull from `cat_search.best_params_`, matching the pattern already used for LightGBM and XGBoost.
- **CatBoost iteration underfitting**: the tuned model was hitting its iteration ceiling (25,000) before early stopping could trigger. Ceiling raised to 30,000 after confirming the model was cut off, not converged.

## Hyperparameter Tuning Methodology

- `RandomizedSearchCV` (`cv=2`, subsampled training data) for the search phase, refit on full `X_train` to find the true `best_iteration`, then a final full-data refit for submission — a deliberate two-phase process, not automated (Optuna) tuning.
- Grid boundaries were extended based on **edge-value analysis**: when a search consistently selected the minimum or maximum value in a parameter's grid, that grid was widened in that direction (e.g., CatBoost's `depth` and `l2_leaf_reg` extended lower, LightGBM's `num_leaves` extended higher).
- Learning rate grids tightened to `[0.0275, 0.03, 0.0325]` after all three models independently converged on 0.03.

## What Didn't Work (Reported Honestly)

- A rolling-average ratio feature (180d / 365d) was tested and added but does not rank among top features in importance — kept since it costs nothing, reported honestly as a negative result.
- Several other engineered features (residual-based features, regional target encoding) were tested and discarded after no measurable leaderboard improvement.

## Still On the Horizon

- Seed-bagging the final three models (training with multiple random seeds and averaging) — the highest-value remaining lever, not yet implemented.
- A windowed version of the base-class prior-mean feature, generalizing the existing rolling-average infrastructure — deliberately deferred until after the viva.

## Environment

- Development: Google Colab (T4 GPU)
- Submission: CPU-only Kaggle kernel (`FinalNotebook`)
- Libraries: `catboost`, `lightgbm`, `xgboost`, `scikit-learn`, `pandas`, `numpy`
