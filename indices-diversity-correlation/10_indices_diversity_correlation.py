"""
Script 10: Acoustic Indices vs Biodiversity Correlation
=========================================================
Correlates acoustic indices (ADI, ACI, AEI, NDSI, MFC, CLS) with
ground-truth biodiversity metrics (Shannon Index, Simpson Index).

Method:
  - Per audio file: compute Shannon/Simpson from BirdNET species detections
  - Merge with acoustic indices computed from same file
  - Bootstrap Spearman correlation with 95% CI

Paper Reference: Section 4.3.3 (acoustic indices interpretation),
                 Part 5 of graphs_analysis notebook
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import os
import sys

# =============================================================================
# CONFIGURATION (imported from 00_config.py)
# =============================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00_config")

INDICES_TO_TEST = ['ADI', 'ACI', 'AEI', 'NDSI', 'MFC', 'CLS']
CONFIDENCE_THRESHOLD = 0.5
N_BOOTSTRAP = 1000

# =============================================================================
# DIVERSITY FUNCTIONS
# =============================================================================
def compute_shannon(labels):
    """Shannon Diversity Index (base-2 entropy of species proportions)."""
    counts = labels.value_counts()
    proportions = counts / len(labels)
    proportions = proportions[proportions > 0]
    return -np.sum(proportions * np.log2(proportions))


def compute_simpson(labels):
    """Simpson's Diversity Index (1 - D)."""
    counts = labels.value_counts()
    n = len(labels)
    if n < 2:
        return np.nan
    numerator = np.sum(counts * (counts - 1))
    denominator = n * (n - 1)
    return 1 - (numerator / denominator)


def bootstrap_spearman(df, col_x, col_y, n_iterations=1000, random_seed=42):
    """Bootstrap Spearman correlation with 95% CI."""
    np.random.seed(random_seed)
    r_values = []
    for _ in range(n_iterations):
        sample = df.sample(frac=1, replace=True)
        r, _ = spearmanr(sample[col_x], sample[col_y])
        r_values.append(r)
    r_values = np.array(r_values)
    return np.mean(r_values), np.percentile(r_values, 2.5), np.percentile(r_values, 97.5)


# =============================================================================
# CORE ANALYSIS
# =============================================================================
def run_analysis(detections_csv, indices_csv, output_dir):
    """Run indices-diversity correlation analysis."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    detections_df = pd.read_csv(detections_csv)
    print(f"  Detections: {len(detections_df)} rows")

    if os.path.exists(indices_csv):
        acoustic_df = pd.read_csv(indices_csv)
        print(f"  Acoustic indices: {len(acoustic_df)} rows")
    else:
        print(f"  WARNING: {indices_csv} not found.")
        print("  Expected columns: filename, ADI, ACI, AEI, NDSI, MFC, CLS")
        acoustic_df = pd.DataFrame()

    if acoustic_df.empty:
        print("\nCannot proceed without acoustic indices data. Exiting.")
        return

    # Normalize column names
    if 'Filename' in detections_df.columns:
        detections_df.rename(columns={'Filename': 'filename'}, inplace=True)
    if 'Filename' in acoustic_df.columns:
        acoustic_df.rename(columns={'Filename': 'filename'}, inplace=True)

    # --- Compute per-file diversity ---
    print("\nComputing per-file biodiversity indices...")
    diversity_df = (
        detections_df[detections_df['confidence'] >= CONFIDENCE_THRESHOLD]
        .groupby('filename')
        .agg(
            Shannon=('label', compute_shannon),
            Simpson=('label', compute_simpson),
            Spot=('Spot', 'first'),
            Date=('Date', 'first')
        )
        .reset_index()
        .dropna(subset=['Shannon', 'Simpson'])
    )
    print(f"  Files with valid diversity: {len(diversity_df)}")

    avg_acoustic = (
        acoustic_df.groupby('filename')[INDICES_TO_TEST].mean().reset_index()
    )

    combined_df = pd.merge(diversity_df, avg_acoustic, on='filename', how='inner')
    print(f"  Merged records: {len(combined_df)}")

    if combined_df.empty:
        print("\nNo matching filenames between detections and indices. Check data.")
        return

    # --- Bootstrap Spearman correlations ---
    print(f"\nBootstrap Spearman correlations (n={N_BOOTSTRAP})...")

    print("\n--- Correlations with Shannon Index ---")
    corr_results_shannon = []
    for index in INDICES_TO_TEST:
        if index in combined_df.columns:
            mean_r, ci_lower, ci_upper = bootstrap_spearman(combined_df, index, 'Shannon', N_BOOTSTRAP)
            corr_results_shannon.append({
                'Index': index, 'Mean_r': mean_r,
                'CI_lower': ci_lower, 'CI_upper': ci_upper
            })
            print(f"  {index}: r={mean_r:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

    print("\n--- Correlations with Simpson Index ---")
    corr_results_simpson = []
    for index in INDICES_TO_TEST:
        if index in combined_df.columns:
            mean_r, ci_lower, ci_upper = bootstrap_spearman(combined_df, index, 'Simpson', N_BOOTSTRAP)
            corr_results_simpson.append({
                'Index': index, 'Mean_r': mean_r,
                'CI_lower': ci_lower, 'CI_upper': ci_upper
            })
            print(f"  {index}: r={mean_r:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

    # --- Forest Plot ---
    df_corr_shannon = pd.DataFrame(corr_results_shannon).sort_values('Mean_r')

    plt.figure(figsize=(10, 7))
    plt.errorbar(
        df_corr_shannon['Mean_r'], df_corr_shannon['Index'],
        xerr=[
            df_corr_shannon['Mean_r'] - df_corr_shannon['CI_lower'],
            df_corr_shannon['CI_upper'] - df_corr_shannon['Mean_r']
        ],
        fmt='o', color='darkslateblue', capsize=5, elinewidth=2, markeredgewidth=2
    )
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("Spearman's Rank Correlation (rs) with Shannon Index", fontsize=12)
    plt.ylabel("Acoustic Index", fontsize=12)
    plt.title("Correlation of Acoustic Indices with Avian Diversity (Shannon)", fontsize=14)
    plt.xlim(-1, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    outpath = os.path.join(output_dir, "correlation_shannon_forest_plot.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"\nSaved: {outpath}")

    # --- Regression Analysis ---
    print("\n--- Regression: Predicting Shannon from Acoustic Indices ---")

    features = [idx for idx in INDICES_TO_TEST if idx in combined_df.columns]
    target = 'Shannon'
    model_df = combined_df.dropna(subset=[target] + features)

    if len(model_df) > 20:
        X = model_df[features]
        y = model_df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        from sklearn.model_selection import GridSearchCV
        print("  Running Random Forest GridSearchCV...")
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_leaf': [3, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='r2', n_jobs=-1)
        grid_search_rf.fit(X_train, y_train)
        best_rf = grid_search_rf.best_estimator_
        print(f"  RF Train R2 = {r2_score(y_train, best_rf.predict(X_train)):.4f}")
        print(f"  RF Test R2  = {r2_score(y_test, best_rf.predict(X_test)):.4f}")

        print("  Running Gradient Boosting GridSearchCV...")
        param_grid_gb = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
        gb = GradientBoostingRegressor(random_state=42)
        grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb, cv=5, scoring='r2', n_jobs=-1)
        grid_search_gb.fit(X_train, y_train)
        best_gb = grid_search_gb.best_estimator_
        print(f"  GB Train R2 = {r2_score(y_train, best_gb.predict(X_train)):.4f}")
        print(f"  GB Test R2  = {r2_score(y_test, best_gb.predict(X_test)):.4f}")
    else:
        print("  Insufficient data for regression analysis.")

    combined_df.to_csv(os.path.join(output_dir, "combined_indices_diversity.csv"), index=False)
    print(f"\nAll results saved to: {output_dir}")


# =============================================================================
# ENTRY POINT — auto-detect mode
# =============================================================================

def _resolve_dependency_aggregate(args, dep_filename):
    """Find a dependency aggregate CSV. Priority: sibling of --aggregate-file > project DB."""
    if args.aggregate_file:
        agg_dir = os.path.dirname(args.aggregate_file)
        candidate = os.path.join(agg_dir, dep_filename)
        if os.path.exists(candidate):
            return candidate
    return None


def _run_watcher_mode():
    """Watcher mode: inline-filter birdnet data + load acoustic indices.

    This script needs BOTH filtered detections and acoustic_indices.
    """
    args = config.parse_common_args(description="10 – Indices vs Diversity (watcher)")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Detections: inline-filter from birdnet aggregate ---
    det_df = config.load_filtered_detections(args)
    if det_df.empty:
        print("ERROR: No data after filtering. Run BirdNET inference first.")
        sys.exit(1)
    det_csv = os.path.join(args.output_dir, "_filtered_detections.csv")
    det_df.to_csv(det_csv, index=False)

    # --- Resolve indices input (still from acoustic_indices aggregate) ---
    idx_agg_path = _resolve_dependency_aggregate(args, "acoustic_indices.csv")
    if idx_agg_path:
        print(f"Loading indices from aggregate: {idx_agg_path}")
        idx_df = config.load_aggregate(idx_agg_path)
        idx_df = config.filter_aggregate_for_output(idx_df, args.start_date, args.end_date, args.spots)
        if idx_df.empty:
            print("ERROR: Indices aggregate is empty after filtering.")
            sys.exit(1)
        idx_csv = os.path.join(args.output_dir, "_filtered_indices.csv")
        idx_df.to_csv(idx_csv, index=False)
    else:
        idx_csv = config.resolve_indices_csv(args)
        if not idx_csv:
            print("ERROR: No acoustic indices CSV found. Run 05 first.")
            sys.exit(1)

    run_analysis(det_csv, idx_csv, args.output_dir)

    # Clean up temp files
    for tmp in ["_filtered_detections.csv", "_filtered_indices.csv"]:
        tmp_path = os.path.join(args.output_dir, tmp)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    config.save_processed_list(args.output_dir, ["aggregate"])


if __name__ == "__main__":
    _run_watcher_mode()
