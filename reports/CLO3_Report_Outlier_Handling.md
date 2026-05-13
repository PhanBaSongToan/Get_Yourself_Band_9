# CLO3: Outlier Detection & Handling

## LLM Semantic Router — Mini Capstone Project

**Student:** Phan Ba Song Toan
**CLO:** CLO3 — Use a variety of techniques for detecting and dealing with outliers
**Input:** `data/processed/engineered_features.csv` (17,708 records)
**Output:** `data/processed/cleaned_features.csv` (14,875 records)

---

## 1. Introduction to Outliers

### 1.1 What is an Outlier?

An **outlier** is a data point that deviates significantly from the majority of observations in a dataset. In the context of text prompt classification, outliers are prompts whose feature values fall far outside the normal distribution — for example, extremely short inputs (single characters), excessively long essays, or spam-like text with repetitive characters.

### 1.2 Why Outliers Affect ML Model Training

Outliers can degrade model performance in several ways:

| Problem | Description |
|---------|-------------|
| **Skewed decision boundaries** | Classifiers like Logistic Regression and SVM try to separate classes with a hyperplane. Extreme values pull the boundary away from the true separation point. |
| **Inflated variance** | Feature scaling (e.g., StandardScaler) uses mean and standard deviation — outliers inflate both, compressing the majority of data into a narrow range. |
| **Misleading feature importance** | Tree-based models (Random Forest, XGBoost) may learn splits based on extreme values that do not generalize to new data. |
| **Corrupted TF-IDF signals** | Very short prompts (1–2 words) produce TF-IDF vectors dominated by a single term, creating misleading signal. |

### 1.3 Detection Approach

We employ two complementary techniques:

1. **Rule-Based Detection** — Domain-specific heuristics derived from knowledge of what constitutes a valid user prompt. These catch semantically invalid inputs that statistical methods might miss.

2. **IQR Statistical Detection** — A distribution-based method that identifies numerically extreme values without assuming a specific distribution (unlike z-score which assumes normality).

---

## 2. Rule-Based Outlier Detection

Rule-based detection applies domain-specific knowledge about text prompt quality. Each rule targets a specific type of invalid or low-quality input.

### Rule 1: `word_count < 3`

**Rationale:** Prompts with fewer than 3 words lack sufficient semantic content for meaningful feature extraction. A 1–2 word input like "Hello" or "OK thanks" cannot produce reliable readability scores, TF-IDF vectors, or complexity assessments.

**Code:**
```python
mask_short = df["word_count"] < 3
```

**Result:** 384 records flagged.

### Rule 2: `char_entropy < 2.5`

**Rationale:** Shannon entropy below 2.5 bits indicates highly repetitive character patterns. Genuine natural language text typically has entropy of 3.5–5.0 bits. Low-entropy inputs are often spam, keyboard mashing (e.g., "aaaaaaaaa"), or copy-pasted filler that does not represent a legitimate user prompt.

**Code:**
```python
mask_entropy = df["char_entropy"] < 2.5
```

**Result:** 234 records flagged.

### Rule 3: `punctuation_ratio > 0.5 AND word_count < 5`

**Rationale:** A prompt where more than half of all characters are punctuation *and* fewer than 5 words exist is almost certainly malformed or non-linguistic input (e.g., "???!!!", "@#$%^&*", "..."). These are not genuine prompts requiring tier classification.

**Code:**
```python
mask_malformed = (df["punctuation_ratio"] > 0.5) & (df["word_count"] < 5)
```

**Result:** 9 records flagged.

### Combined Rule-Based Results

Rules are combined using a union (OR) to avoid double-counting records that match multiple rules:

```python
combined_mask = mask_short | mask_entropy | mask_malformed
df_clean = df[~combined_mask].copy()
```

| Rule | Records Flagged |
|------|:---:|
| word_count < 3 | 384 |
| char_entropy < 2.5 | 234 |
| punct > 0.5 & words < 5 | 9 |
| **Combined (union)** | **394** |

---

## 3. Statistical Outlier Detection (IQR Method)

### 3.1 What is the IQR Method?

The **Interquartile Range (IQR)** method identifies outliers based on the spread of the middle 50% of the data. It is robust to non-normal distributions, making it ideal for text features that are typically right-skewed.

**Formula:**

$$IQR = Q_3 - Q_1$$
$$\text{Lower Bound} = Q_1 - 1.5 \times IQR$$
$$\text{Upper Bound} = Q_3 + 1.5 \times IQR$$

Any value falling below the lower bound or above the upper bound is classified as an outlier. The multiplier of **1.5** is the standard threshold for moderate outlier detection (as opposed to 3.0 for extreme outliers only).

### 3.2 Why IQR Was Chosen Over Z-Score

| Criterion | Z-Score | IQR |
|-----------|---------|-----|
| Assumes normality | Yes | No |
| Robust to skewed data | No | **Yes** |
| Affected by extreme values | Yes (mean/std shift) | **No** (quartiles are resistant) |
| Suitable for text features | Limited | **Ideal** |

Text features like `prompt_length` and `word_count` follow a right-skewed distribution (many short prompts, few very long ones). IQR handles this correctly without being distorted by the long tail.

### 3.3 IQR Results

The IQR method was applied to `prompt_length` and `word_count` on the dataset *after* rule-based cleaning (17,314 records):

| Feature | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outliers |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| prompt_length | 41.0 | 148.0 | 107.0 | -119.5 | 308.5 | 2,327 |
| word_count | 8.0 | 26.0 | 18.0 | -19.0 | 53.0 | 2,305 |

**Note:** The negative lower bounds (e.g., -119.5) are below zero and therefore have no practical effect — no prompt can have negative length. Only the upper bounds are active filters in this case.

**Code:**
```python
Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
```

**Combined IQR outliers (union of both features):** 2,439 records removed.

---

## 4. Results Summary

### 4.1 Before vs. After Comparison

| Stage | Records | Removed | Cumulative Removed |
|-------|:---:|:---:|:---:|
| Initial dataset | 17,708 | — | — |
| After rule-based cleaning | 17,314 | 394 | 394 (2.22%) |
| After IQR cleaning | 14,875 | 2,439 | 2,833 (16.00%) |

### 4.2 Tier Distribution After Cleaning

| Target Tier | Count | Percentage |
|-------------|:---:|:---:|
| Tier_2_API | 7,732 | 52.0% |
| Tier_1_Local | 7,143 | 48.0% |

The class balance remains healthy after outlier removal (ratio ≈ 1.08x), confirming that the cleaning process did not disproportionately affect one class.

### 4.3 Output File

| File | Description | Size |
|------|-------------|------|
| `data/processed/cleaned_features.csv` | Cleaned feature matrix (14,875 rows × 1,009 columns) | 58.7 MB |

---

## 5. Conclusion

### 5.1 Summary

Two complementary outlier detection techniques were applied to the engineered feature dataset:

1. **Rule-based detection** removed 394 records (2.22%) that were semantically invalid — too short, too repetitive, or structurally malformed to represent genuine user prompts.

2. **IQR statistical detection** removed 2,439 additional records (14.09% of the remaining data) that had extreme `prompt_length` or `word_count` values falling outside 1.5× IQR bounds.

In total, **2,833 records (16.00%)** were removed, reducing the dataset from 17,708 to **14,875 records**.

### 5.2 Readiness for CLO4

The cleaned dataset (`cleaned_features.csv`) is now ready for model training in CLO4. Key properties:
- **14,875 records** — sufficient for training and evaluation splits
- **Balanced classes** — Tier_2_API (52.0%) vs. Tier_1_Local (48.0%)
- **1,008 features** — 8 hand-crafted + 1,000 TF-IDF
- **No extreme outliers** that could distort model learning

### 5.3 Caveats

- Some prompts near the IQR boundary (e.g., 300–310 characters) are borderline cases. They were removed conservatively to improve model generalization, but could be retained in future iterations with domain expert review.
- The IQR method primarily removed long, verbose prompts. If the production use case expects many long prompts, the upper bound threshold could be relaxed to 2.0× or 3.0× IQR.
- Rule-based thresholds (e.g., `char_entropy < 2.5`) were chosen based on distribution analysis and may need adjustment if the dataset composition changes.

---

*Report generated for CLO3 — Outlier Detection & Handling*
*LLM Semantic Router — Mini Capstone Project*
