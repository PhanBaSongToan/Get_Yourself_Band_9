"""
=============================================================================
 CLO3 -- OUTLIER DETECTION & HANDLING
 LLM Semantic Router -- Mini Capstone Project
=============================================================================

## Purpose
This script identifies and removes outlier records from the engineered
features dataset produced by CLO2. Outliers are records whose feature
values are so extreme that they would distort model training in CLO4.

## Outlier Detection Strategy

### 1. Rule-Based Detection (domain-specific)
Three rules rooted in domain knowledge about text prompt quality:
  - word_count < 3            -> too short for meaningful semantic analysis
  - char_entropy < 2.5        -> repetitive / spam-like character patterns
  - punctuation_ratio > 0.5
    AND word_count < 5         -> malformed input (e.g. "???!!!", "@#$%")

### 2. Statistical Detection (IQR method)
The Interquartile Range method flags values beyond 1.5x IQR from the
quartile boundaries as statistical outliers.  Applied to:
  - prompt_length
  - word_count

## Inputs
  - data/processed/engineered_features.csv  (17,708 rows x 1009 columns)

## Outputs
  - data/processed/cleaned_features.csv     (cleaned dataset)

=============================================================================
"""

# -- Standard Library --------------------------------------------------------
import logging
import os
import sys
import time
from pathlib import Path

# -- Third-Party -------------------------------------------------------------
import pandas as pd
import numpy as np

# -- Fix Windows console encoding --------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# =============================================================================
#  LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SemanticRouter.OutlierHandling")


# =============================================================================
#  PATH CONFIGURATION
# =============================================================================
# Resolve paths relative to the project root (two levels up from this script).
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent.parent

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "engineered_features.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "cleaned_features.csv"


# =============================================================================
#  HELPER: Console Separator
# =============================================================================
def separator(title: str = "", char: str = "-", width: int = 65) -> None:
    """Print a titled separator line for readable console output."""
    print(f"\n  {char * width}")
    if title:
        print(f"  {title}")
    print(f"  {char * width}")


# =============================================================================
#  RULE-BASED OUTLIER DETECTION
# =============================================================================
def apply_rule_based_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-specific rules to flag and remove known outlier patterns.

    Rules
    -----
    1. word_count < 3
       Prompts with fewer than 3 words lack sufficient semantic content
       for meaningful TF-IDF or readability features. Examples: "Hi", "OK",
       single-token inputs.

    2. char_entropy < 2.5
       Shannon entropy below 2.5 bits indicates highly repetitive character
       patterns -- typically spam, keyboard mashing, or copy-pasted
       filler text that does not represent a genuine user prompt.

    3. punctuation_ratio > 0.5 AND word_count < 5
       A high punctuation ratio combined with very few words signals
       malformed or non-linguistic input (e.g., "???!!!", "@#$%^&*").

    Parameters
    ----------
    df : pd.DataFrame
        The feature-engineered dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with rule-based outliers removed.
    """
    initial_count = len(df)
    logger.info("Starting rule-based outlier detection on %d records", initial_count)

    # ---- Rule 1: word_count < 3 ----
    mask_short = df["word_count"] < 3
    n_short = mask_short.sum()
    logger.info("Rule 1 | word_count < 3           : %d records flagged", n_short)

    # ---- Rule 2: char_entropy < 2.5 ----
    mask_entropy = df["char_entropy"] < 2.5
    n_entropy = mask_entropy.sum()
    logger.info("Rule 2 | char_entropy < 2.5        : %d records flagged", n_entropy)

    # ---- Rule 3: punctuation_ratio > 0.5 AND word_count < 5 ----
    mask_malformed = (df["punctuation_ratio"] > 0.5) & (df["word_count"] < 5)
    n_malformed = mask_malformed.sum()
    logger.info("Rule 3 | punct > 0.5 & words < 5   : %d records flagged", n_malformed)

    # ---- Combine all rule masks (union) ----
    combined_mask = mask_short | mask_entropy | mask_malformed
    total_flagged = combined_mask.sum()

    df_clean = df[~combined_mask].copy().reset_index(drop=True)

    logger.info(
        "Rule-based total removed: %d records (%.2f%%). Remaining: %d",
        total_flagged, total_flagged / initial_count * 100, len(df_clean),
    )

    # Print console summary
    print(f"\n  Rule-Based Outlier Detection Results:")
    print(f"    Rule 1 (word_count < 3)              : {n_short:>6} flagged")
    print(f"    Rule 2 (char_entropy < 2.5)          : {n_entropy:>6} flagged")
    print(f"    Rule 3 (punct > 0.5 & words < 5)     : {n_malformed:>6} flagged")
    print(f"    Combined (union, no double-counting)  : {total_flagged:>6} removed")
    print(f"    Remaining records                     : {len(df_clean):>6}")

    return df_clean


# =============================================================================
#  STATISTICAL OUTLIER DETECTION (IQR METHOD)
# =============================================================================
def apply_iqr_filter(df: pd.DataFrame, column: str, multiplier: float = 1.5):
    """
    Compute IQR bounds and flag outliers for a single column.

    The IQR method defines outliers as observations that fall below
    Q1 - k*IQR or above Q3 + k*IQR, where k is the multiplier
    (standard value = 1.5).

    Parameters
    ----------
    df : pd.DataFrame
        The current dataset.
    column : str
        Column name to apply IQR detection on.
    multiplier : float
        IQR multiplier (default 1.5 for standard outlier detection).

    Returns
    -------
    pd.Series (bool)
        Boolean mask where True = outlier.
    dict
        Dictionary with IQR statistics for reporting.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)

    stats = {
        "column": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outliers": outlier_mask.sum(),
    }

    logger.info(
        "IQR [%s] | Q1=%.1f  Q3=%.1f  IQR=%.1f  "
        "bounds=[%.1f, %.1f]  outliers=%d",
        column, Q1, Q3, IQR, lower_bound, upper_bound, outlier_mask.sum(),
    )

    return outlier_mask, stats


def apply_statistical_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply IQR-based outlier detection on prompt_length and word_count.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset after rule-based cleaning.

    Returns
    -------
    pd.DataFrame
        Dataset with IQR outliers removed.
    """
    initial_count = len(df)
    logger.info("Starting IQR outlier detection on %d records", initial_count)

    # ---- IQR on prompt_length ----
    mask_pl, stats_pl = apply_iqr_filter(df, "prompt_length")

    # ---- IQR on word_count ----
    mask_wc, stats_wc = apply_iqr_filter(df, "word_count")

    # ---- Combine both IQR masks (union) ----
    combined_iqr = mask_pl | mask_wc
    total_iqr = combined_iqr.sum()

    df_clean = df[~combined_iqr].copy().reset_index(drop=True)

    logger.info(
        "IQR total removed: %d records (%.2f%%). Remaining: %d",
        total_iqr, total_iqr / initial_count * 100, len(df_clean),
    )

    # Print console summary
    print(f"\n  IQR Statistical Outlier Detection Results:")
    print(f"  +----------------+---------+---------+---------+----------+----------+----------+")
    print(f"  | Feature        |      Q1 |      Q3 |     IQR |    Lower |    Upper | Outliers |")
    print(f"  +----------------+---------+---------+---------+----------+----------+----------+")
    for s in [stats_pl, stats_wc]:
        print(
            f"  | {s['column']:<14} | {s['Q1']:>7.1f} | {s['Q3']:>7.1f} | {s['IQR']:>7.1f} "
            f"| {s['lower_bound']:>8.1f} | {s['upper_bound']:>8.1f} | {s['outliers']:>8} |"
        )
    print(f"  +----------------+---------+---------+---------+----------+----------+----------+")
    print(f"    Combined IQR outliers (union) : {total_iqr:>6} removed")
    print(f"    Remaining records             : {len(df_clean):>6}")

    return df_clean


# =============================================================================
#  MAIN PIPELINE
# =============================================================================
def main() -> None:
    """Execute the complete CLO3 outlier detection and handling pipeline."""
    pipeline_start = time.time()

    separator("CLO3 -- OUTLIER DETECTION & HANDLING PIPELINE", char="=")
    logger.info("Pipeline started")

    # =========================================================================
    #  STAGE 1 -- Load Data
    # =========================================================================
    separator("STAGE 1 -- Loading engineered features")

    if not INPUT_CSV.exists():
        logger.error("Input file not found: %s", INPUT_CSV)
        sys.exit(1)

    logger.info("Loading %s", INPUT_CSV)
    df = pd.read_csv(str(INPUT_CSV))
    initial_count = len(df)

    logger.info("Loaded %d records with %d columns", len(df), len(df.columns))
    print(f"  Input file    : {INPUT_CSV}")
    print(f"  Initial rows  : {initial_count:,}")
    print(f"  Columns       : {len(df.columns)}")

    # =========================================================================
    #  STAGE 2 -- Rule-Based Outlier Detection
    # =========================================================================
    separator("STAGE 2 -- Rule-Based Outlier Detection")
    df_after_rules = apply_rule_based_filters(df)
    rule_removed = initial_count - len(df_after_rules)

    # =========================================================================
    #  STAGE 3 -- Statistical Outlier Detection (IQR)
    # =========================================================================
    separator("STAGE 3 -- Statistical Outlier Detection (IQR)")
    df_final = apply_statistical_filters(df_after_rules)
    iqr_removed = len(df_after_rules) - len(df_final)

    # =========================================================================
    #  STAGE 4 -- Save Cleaned Dataset
    # =========================================================================
    separator("STAGE 4 -- Saving cleaned dataset")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(str(OUTPUT_CSV), index=False)
    logger.info("Cleaned dataset saved -> %s", OUTPUT_CSV)
    print(f"  Output file : {OUTPUT_CSV}")
    print(f"  Final rows  : {len(df_final):,}")
    print(f"  File size   : {OUTPUT_CSV.stat().st_size / (1024*1024):.1f} MB")

    # =========================================================================
    #  STAGE 5 -- Summary Report
    # =========================================================================
    separator("STAGE 5 -- Summary Report")

    total_removed = initial_count - len(df_final)
    pct_removed = total_removed / initial_count * 100

    print(f"  +---------------------------------------------+")
    print(f"  |  CLO3 Outlier Handling -- Summary            |")
    print(f"  +---------------------+---------+--------------+")
    print(f"  | Stage               | Records | Removed      |")
    print(f"  +---------------------+---------+--------------+")
    print(f"  | Initial             | {initial_count:>7,} | -            |")
    print(f"  | After rule-based    | {len(df_after_rules):>7,} | {rule_removed:>5,}        |")
    print(f"  | After IQR           | {len(df_final):>7,} | {iqr_removed:>5,}        |")
    print(f"  +---------------------+---------+--------------+")
    print(f"  | Total removed       |    -    | {total_removed:>5,} ({pct_removed:.2f}%) |")
    print(f"  +---------------------+---------+--------------+")

    # Tier distribution after cleaning
    tier_counts = df_final["Target_Tier"].value_counts()
    print(f"\n  Tier distribution after cleaning:")
    for tier, count in tier_counts.items():
        pct = count / len(df_final) * 100
        print(f"    {tier:<14}  {count:>6,}  ({pct:.1f}%)")

    elapsed = time.time() - pipeline_start
    separator("CLO3 -- PIPELINE COMPLETE", char="=")
    logger.info("Pipeline completed in %.1f seconds", elapsed)
    print(f"  Total time: {elapsed:.1f} seconds\n")


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
