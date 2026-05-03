"""
=============================================================================
 CLO2 — FEATURE SELECTION & ENGINEERING
 LLM Semantic Router — Mini Capstone Project
=============================================================================

## Purpose
This script transforms raw user prompts stored in `router_logs.db` into a
rich numerical feature matrix suitable for training a binary classifier that
routes prompts to either:
  - Tier_1_Local  (simple prompts → small local model)
  - Tier_2_API    (complex prompts → large cloud model like GPT-4 / Claude)

## Feature Categories

### 1. Statistical Features (5)
Surface-level text statistics that capture prompt length and lexical diversity:
  • prompt_length     — total character count
  • word_count        — total number of words
  • avg_word_len      — mean word length in characters
  • unique_word_ratio — type-token ratio (lexical diversity)
  • punctuation_ratio — proportion of punctuation characters

### 2. Semantic Features (3)
Deeper linguistic signals that capture cognitive complexity and readability:
  • complex_keyword_count — count of Bloom's Taxonomy higher-order keywords
  • char_entropy          — Shannon entropy of character distribution
  • flesch_reading_ease   — Flesch Reading Ease score (textstat library)

### 3. Text Vectorization (1000 TF-IDF features)
Bag-of-words representation using TF-IDF with unigrams + bigrams, capturing
the actual vocabulary patterns that distinguish simple from complex prompts.

## Outputs
  • engineered_features.csv   — 1008 columns (8 hand-crafted + 1000 TF-IDF)
  • tfidf_vectorizer.pkl      — fitted TfidfVectorizer (reused in CLO9 serving)

=============================================================================
"""

# ── Standard Library ─────────────────────────────────────────────────────────
import logging
import math
import os
import re
import string
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

# ── Third-Party ──────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import joblib
import textstat

# ── Fix Windows console encoding (cp1252 cannot print Unicode) ───────────────
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # non-Windows or already UTF-8
from sklearn.feature_extraction.text import TfidfVectorizer

# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SemanticRouter.FeatureEngineering")


# ═══════════════════════════════════════════════════════════════════════════════
#  PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# All paths are resolved relative to this script's location so the script
# works regardless of the current working directory.
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = SCRIPT_DIR / "data"
DB_PATH = DATA_DIR / "router_logs.db"

# Output artifacts (saved alongside the script for easy access)
OUTPUT_CSV = SCRIPT_DIR / "engineered_features.csv"
TFIDF_PKL = SCRIPT_DIR / "tfidf_vectorizer.pkl"


# ═══════════════════════════════════════════════════════════════════════════════
#  BLOOM'S TAXONOMY HIGHER-ORDER KEYWORDS
# ═══════════════════════════════════════════════════════════════════════════════
# These keywords signal cognitive complexity — prompts containing more of
# these words are more likely to require a powerful Tier 2 model.
# Source: Bloom's Revised Taxonomy (Anderson & Krathwohl, 2001)
BLOOM_KEYWORDS = [
    "explain", "compare", "analyze", "evaluate", "synthesize",
    "prove", "debug", "implement", "translate", "summarize",
    "optimize", "design", "justify", "critique", "develop",
    "differentiate", "interpret", "predict", "calculate",
]

# Pre-compile a single regex for efficient matching (word boundaries, case-insensitive)
_BLOOM_PATTERN = re.compile(
    r"\b(?:" + "|".join(BLOOM_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Statistical Features ─────────────────────────────────────────────────────

def compute_prompt_length(text: str) -> int:
    """Total number of characters in the prompt."""
    return len(text)


def compute_word_count(text: str) -> int:
    """Number of whitespace-delimited words."""
    return len(text.split())


def compute_avg_word_len(text: str) -> float:
    """Average character length per word. Returns 0.0 for empty prompts."""
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def compute_unique_word_ratio(text: str) -> float:
    """
    Type-Token Ratio (TTR): unique words / total words.
    Higher ratio → more diverse vocabulary → potentially harder prompt.
    Returns 0.0 for empty prompts.
    """
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_punctuation_ratio(text: str) -> float:
    """
    Proportion of characters that are punctuation.
    Prompts with code or mathematical notation tend to have higher ratios.
    Returns 0.0 for empty prompts.
    """
    if not text:
        return 0.0
    punct_count = sum(1 for ch in text if ch in string.punctuation)
    return punct_count / len(text)


# ── Semantic Features ────────────────────────────────────────────────────────

def compute_complex_keyword_count(text: str) -> int:
    """
    Count of Bloom's Taxonomy higher-order cognitive keywords found in text.
    Uses pre-compiled regex for efficiency across 17K+ rows.
    """
    return len(_BLOOM_PATTERN.findall(text))


def compute_char_entropy(text: str) -> float:
    """
    Shannon entropy of the character distribution.

    Formula: H = -Σ p(c) · log₂(p(c))

    High entropy → diverse character usage → potentially meaningful text.
    Low entropy  → repetitive characters  → spam or trivial prompt.
    Returns 0.0 for empty prompts.
    """
    if not text:
        return 0.0
    freq = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_flesch_reading_ease(text: str) -> float:
    """
    Flesch Reading Ease score via the `textstat` library.

    Score interpretation:
      90–100 → Very easy  (5th grade)
      60–70  → Standard   (8th–9th grade)
      0–30   → Very hard  (college graduate)

    Lower scores → harder text → more likely Tier_2_API.
    """
    try:
        return textstat.flesch_reading_ease(text)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def separator(title: str = "", char: str = "-", width: int = 65) -> None:
    """Print a titled separator line for readable console output."""
    print(f"\n  {char * width}")
    if title:
        print(f"  {title}")
    print(f"  {char * width}")


def main() -> None:
    """Execute the complete CLO2 feature engineering pipeline."""
    pipeline_start = time.time()

    separator("CLO2 -- FEATURE ENGINEERING PIPELINE START", char="=")
    logger.info("Pipeline started")

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 1 — Load data from SQLite
    # ══════════════════════════════════════════════════════════════════════════
    separator("STAGE 1 — Loading data from SQLite")

    if not DB_PATH.exists():
        logger.error("Database not found at %s", DB_PATH)
        sys.exit(1)

    logger.info("Connecting to database → %s", DB_PATH)
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query("SELECT id, user_prompt, Target_Tier FROM historical_prompts", conn)
    conn.close()

    logger.info("Loaded %d rows from 'historical_prompts'", len(df))
    print(f"  Rows loaded  : {len(df):,}")
    print(f"  Columns      : {list(df.columns)}")
    print(f"\n  First 5 prompts:")
    for i, row in df.head(5).iterrows():
        print(f"    [{i}] ({row['Target_Tier']:>12}) {row['user_prompt'][:80]}...")

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 2 — Extract Statistical Features
    # ══════════════════════════════════════════════════════════════════════════
    separator("STAGE 2 — Extracting Statistical Features")

    prompts = df["user_prompt"].fillna("").astype(str)

    logger.info("Computing prompt_length...")
    df["prompt_length"] = prompts.apply(compute_prompt_length)

    logger.info("Computing word_count...")
    df["word_count"] = prompts.apply(compute_word_count)

    logger.info("Computing avg_word_len...")
    df["avg_word_len"] = prompts.apply(compute_avg_word_len)

    logger.info("Computing unique_word_ratio...")
    df["unique_word_ratio"] = prompts.apply(compute_unique_word_ratio)

    logger.info("Computing punctuation_ratio...")
    df["punctuation_ratio"] = prompts.apply(compute_punctuation_ratio)

    stat_cols = [
        "prompt_length", "word_count", "avg_word_len",
        "unique_word_ratio", "punctuation_ratio",
    ]

    print("\n  ✅ Statistical features extracted successfully!")
    print(f"\n  First 5 rows of statistical features:")
    print(df[["user_prompt"] + stat_cols].head(5).to_string(index=False))

    # Summary statistics
    print(f"\n  Descriptive statistics:")
    print(df[stat_cols].describe().round(2).to_string())

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 3 — Extract Semantic Features
    # ══════════════════════════════════════════════════════════════════════════
    separator("STAGE 3 — Extracting Semantic Features")

    logger.info("Computing complex_keyword_count (Bloom's Taxonomy)...")
    df["complex_keyword_count"] = prompts.apply(compute_complex_keyword_count)

    logger.info("Computing char_entropy (Shannon entropy)...")
    df["char_entropy"] = prompts.apply(compute_char_entropy)

    logger.info("Computing flesch_reading_ease (textstat)...")
    df["flesch_reading_ease"] = prompts.apply(compute_flesch_reading_ease)

    sem_cols = ["complex_keyword_count", "char_entropy", "flesch_reading_ease"]

    print("\n  ✅ Semantic features extracted successfully!")
    print(f"\n  First 5 rows of semantic features:")
    print(df[["user_prompt"] + sem_cols].head(5).to_string(index=False))

    # Summary statistics
    print(f"\n  Descriptive statistics:")
    print(df[sem_cols].describe().round(2).to_string())

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 4 — TF-IDF Vectorization
    # ══════════════════════════════════════════════════════════════════════════
    separator("STAGE 4 — TF-IDF Vectorization")

    logger.info("Fitting TfidfVectorizer (max_features=1000, ngram_range=(1,2))...")

    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 2),
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(prompts)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    logger.info("TF-IDF matrix shape: %s", tfidf_matrix.shape)
    print(f"  TF-IDF matrix shape : {tfidf_matrix.shape}")
    print(f"  Number of features  : {len(tfidf_feature_names)}")
    print(f"  First 20 features   : {list(tfidf_feature_names[:20])}")
    print(f"  Sparsity            : {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.4%}")

    # Save the fitted vectorizer for CLO9 (inference pipeline)
    joblib.dump(tfidf_vectorizer, str(TFIDF_PKL))
    logger.info("TF-IDF vectorizer saved → %s", TFIDF_PKL)
    print(f"\n  ✅ Vectorizer saved → {TFIDF_PKL}")

    # Convert sparse matrix to DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{name}" for name in tfidf_feature_names],
        index=df.index,
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 5 — Combine All Features & Export
    # ══════════════════════════════════════════════════════════════════════════
    separator("STAGE 5 — Combining Features & Exporting")

    # Combine: statistical (5) + semantic (3) + TF-IDF (1000)
    all_feature_cols = stat_cols + sem_cols
    features_df = pd.concat([
        df[all_feature_cols],  # 8 hand-crafted features
        tfidf_df,              # 1000 TF-IDF features
    ], axis=1)

    # Add Target_Tier label back
    features_df["Target_Tier"] = df["Target_Tier"]

    logger.info("Final feature matrix shape: %s", features_df.shape)
    print(f"  Total hand-crafted features : {len(all_feature_cols)}")
    print(f"  Total TF-IDF features       : {tfidf_matrix.shape[1]}")
    print(f"  Total features              : {features_df.shape[1] - 1}")  # minus Target_Tier
    print(f"  Final matrix shape          : {features_df.shape}")

    # Save to CSV
    features_df.to_csv(str(OUTPUT_CSV), index=False)
    logger.info("Engineered features saved → %s", OUTPUT_CSV)
    print(f"\n  ✅ Features saved → {OUTPUT_CSV}")
    print(f"     File size: {OUTPUT_CSV.stat().st_size / (1024 * 1024):.1f} MB")

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 6 — Feature Summary Report
    # ══════════════════════════════════════════════════════════════════════════
    separator("STAGE 6 — Feature Summary Report")

    print(f"  +-------------------------------------------------------------+")
    print(f"  |  CLO2 Feature Engineering -- Summary                        |")
    print(f"  +-------------------------------------------------------------+")
    print(f"  |  Input rows             :  {len(df):>8,}                        |")
    print(f"  |  Statistical features   :  {len(stat_cols):>8}                        |")
    print(f"  |  Semantic features       :  {len(sem_cols):>7}                        |")
    print(f"  |  TF-IDF features        :  {tfidf_matrix.shape[1]:>8}                        |")
    print(f"  |  Total feature columns  :  {features_df.shape[1] - 1:>8}                        |")
    print(f"  +-------------------------------------------------------------+")
    print(f"  |  Tier distribution in output:                              |")
    tier_counts = features_df["Target_Tier"].value_counts()
    for tier, count in tier_counts.items():
        pct = count / len(features_df) * 100
        print(f"  |    {tier:<14}  {count:>6,}  ({pct:>5.1f}%)                       |")
    print(f"  +-------------------------------------------------------------+")
    print(f"  |  Output files:                                             |")
    print(f"  |    - engineered_features.csv                               |")
    print(f"  |    - tfidf_vectorizer.pkl                                  |")
    print(f"  +-------------------------------------------------------------+")

    elapsed = time.time() - pipeline_start
    separator("CLO2 -- PIPELINE COMPLETE", char="=")
    logger.info("Pipeline completed in %.1f seconds", elapsed)
    print(f"  Total time: {elapsed:.1f} seconds\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
