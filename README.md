# LLM Semantic Router

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-in--progress-yellow.svg)](#course-learning-outcomes-clos-and-progress)

An intelligent machine learning system designed to classify user prompts into distinct tiers, routing them to either local small language models or large cloud-based APIs to optimize for cost, latency, and performance.

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Course Learning Outcomes (CLOs) and Progress](#-course-learning-outcomes-clos-and-progress)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Team Members](#-team-members)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 💡 Problem Statement
As Large Language Models (LLMs) become central to modern applications, organizations face a critical trade-off between **cost** and **capability**. Routing every simple query (e.g., "What time is it?") to a high-cost, high-latency API like GPT-4 is inefficient. 

The **LLM Semantic Router** solves this by acting as a gateway that:
1. **Analyzes** the complexity of incoming prompts.
2. **Routes** "Tier 1" (simple) prompts to lightweight local models.
3. **Escalates** "Tier 2" (complex) prompts to powerful cloud-based APIs.

This approach significantly reduces operational costs while maintaining high-quality responses for complex tasks.

---

## 📊 Dataset
The project utilizes a cleaned version of the **LMSYS Chatbot Arena** dataset.
- **Size**: 17,708 labeled records.
- **Source**: `router_logs.db` (SQLite).
- **Labels**:
    - `Tier_1_Local`: Easy/basic prompts (e.g., simple questions, greetings).
    - `Tier_2_API`: Hard/complex prompts (e.g., coding tasks, complex reasoning, multi-step logic).

---

## 📈 Course Learning Outcomes (CLOs) and Progress

| CLO | Description | Status |
|-----|-------------|--------|
| CLO1 | Data retrieval from SQL, NoSQL, APIs, Cloud | ✅ Done |
| CLO2 | Feature selection & engineering | ✅ Done |
| CLO3 | Outlier detection & handling | ⏳ Next |
| CLO4 | Supervised model training | ⏳ Pending |
| CLO5 | Model evaluation & error metrics | ⏳ Pending |
| CLO6 | Handle unbalanced classes | ⏳ Pending |
| CLO7 | Explain unsupervised learning approaches | ⏳ Pending |
| CLO8 | Explain curse of dimensionality | ⏳ Pending |
| CLO9 | Clustering & dimensionality reduction | ⏳ Pending |
| CLO10 | Mini capstone final report & presentation | ⏳ Pending |

---

## 📂 Project Structure

```text
LLM-Semantic-Router/
├── data/                       # Data storage
│   ├── raw/                    # Raw downloads
│   ├── processed/              # Cleaned/Engineered data
│   └── external/               # Third-party data
├── models/                     # Saved model artifacts (e.g., .pkl files)
├── notebooks/                  # Jupyter notebooks for exploration
├── reports/                    # Generated analysis reports
├── src/                        # Source code
│   ├── data_collection/        # CLO1 scripts
│   ├── features/               # CLO2 feature engineering logic
│   ├── models/                 # Training and inference scripts
│   └── unsupervised/           # Unsupervised learning exploration
├── router_logs.db              # Main SQLite database
├── README.md                   # Project documentation
└── .gitignore                  # Git exclusion rules
```

---

## ⚙️ Installation & Usage

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed.

### 2. Clone and Install
```bash
git clone https://github.com/your-username/LLM-Semantic-Router.git
cd LLM-Semantic-Router
pip install -r requirements.txt
```

### 3. Running CLO Scripts
Each stage of the project can be executed via its respective script:

**CLO1: Data Collection**
```bash
python src/data_collection/clo1_data_collection.py
```

**CLO2: Feature Engineering**
```bash
python src/features/CLO2_feature_engineering.py
```

---

## 👥 Team Members
1. **[Name Placeholder]** - Role/Contribution
2. **[Name Placeholder]** - Role/Contribution
3. **[Name Placeholder]** - Role/Contribution
4. **[Name Placeholder]** - Role/Contribution
5. **[Name Placeholder]** - Role/Contribution

---

## 🤝 Contributing
Contributions are welcome! To ensure consistency, please follow these guidelines:
- **Branch Naming**: Use `feature/cloX-description` or `fix/issue-description`.
- **New CLOs**: Add your script to the appropriate subdirectory in `src/` and update the Progress Table in this README.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ✉️ Contact
**Phan Ba Song Toan**  
GitHub: [@PhanBaSongToan](https://github.com/PhanBaSongToan)
