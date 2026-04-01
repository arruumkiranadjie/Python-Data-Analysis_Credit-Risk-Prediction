# Credit Risk Prediction - Python Data Analysis Project

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Anaconda](https://img.shields.io/badge/Anaconda-44A833?style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)]()
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)]()
[![Markdown](https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white)](https://www.markdownguide.org)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com)

## 📌 Overview

*Python Data Analysis Portfolio - Credit Risk Prediction* delivers an end-to-end machine learning pipeline for predicting borrower credit risk using real loan data spanning **2007–2014** from a Lending Company, covering over **466,285 loan records** across **74 raw features**. The primary objective is to classify borrowers as **Qualified** or **Unqualified** — providing a data-driven, binary decision support tool to assist lending institutions in reducing default exposure and improving credit approval precision.

The project is structured as a complete, production-oriented analytical workflow across four sequential phases:

1. **Data Cleaning:** Resolves missing values, removes zero-information columns, enforces input validation across key financial fields, applies label encoding to categorical and datetime features, and conducts IQR-based outlier detection.
2. **Exploratory Data Analysis (EDA):** Identifies and encodes the target variable `loan_status` into a binary `loan_status_number` label, and examines borrower distribution across geography, repayment behavior, and loan status composition through statistical and visual analysis.
3. **Feature Selection:** Applies Pearson correlation filtering for numerical features (threshold: |r| > 0.1 against `loan_status_number`) and Point-Biserial correlation for categorical features, isolating `grade` and `sub_grade` as the most predictive categorical signals.
4. **Predictive Modeling:** Trains and evaluates two classification models — a **Decision Tree Classifier** (`max_depth=10`) for interpretable rule-based prediction, and a **Random Forest Classifier** (`max_depth=50`, with StandardScaler preprocessing and 80/20 train-test split) for ensemble-based accuracy and feature importance ranking.

## 📂 Repository Structure

```
Python-Data-Analysis_Credit-Risk-Prediction
│
├── README.md                                                        ← README
│
├── Credit_Risk_Prediction-Decision_Tree_Random_Forest.ipynb         ← Main Notebook
│
└── src                                                              ← Source For This Project
    └── data
        └── loan_data_2007_2014.csv                                  ← Loan Dataset (2007–2014)
```

## 📊 Dataset

The dataset consists of real loan application and repayment records from a Lending Company, covering **466,285 borrowers** across **74 original features** before cleaning. It captures the full financial profile of each loan — from application attributes and borrower demographics to repayment timelines and final loan outcomes.

| Column Group | Key Features | Description |
|---|---|---|
| Loan Attributes | `loan_amnt`, `funded_amnt`, `int_rate`, `term`, `grade`, `sub_grade` | Core loan terms, interest rates, and risk grading |
| Borrower Profile | `emp_length`, `home_ownership`, `annual_inc`, `addr_state` | Employment, housing status, income, and geographic location |
| Repayment & Recovery | `total_pymnt`, `total_rec_prncp`, `recoveries`, `last_pymnt_amnt` | Payment history, principal recovered, and collections data |
| Loan Outcome | `loan_status` → `loan_status_number` | Original 9-class status encoded to binary: 1 (Qualified), 0 (Unqualified) |
| Datetime Features | `issue_d`, `last_pymnt_d`, `last_credit_pull_d`, `next_pymnt_d` | Decomposed into separate month and year integer columns |

## 🧠 Analyses

1. **Data Cleaning & Preprocessing:** Reduces the raw 74-column dataset to 48 analytically relevant features by dropping null-dominant columns, free-text fields, and unique identifiers. Enforces non-negativity constraints on financial columns, standardizes `term` and `emp_length` to numeric types, normalizes `home_ownership` categories to the four valid company-defined values, and decomposes all datetime columns into integer month and year pairs.

2. **Target Variable Engineering:** Maps the original 9-value `loan_status` column into a binary classification label: Current, Fully Paid, In Grace Period, and policy-compliant Fully Paid statuses are encoded as **1 (Qualified)**; Charged Off, Default, Late (16–30 days), Late (31–120 days), and policy-non-compliant statuses are encoded as **0 (Unqualified)**. This binary encoding forms the foundation for all downstream modeling.

3. **Exploratory Data Analysis (EDA):** Visualizes the geographic distribution of borrowers across U.S. states via a Plotly choropleth map, identifying California (CA) as the highest-volume applicant state. Comparative bar charts using Seaborn contrast the original 9-class `loan_status` distribution against the derived binary `loan_status_number`, confirming that Qualified borrowers dominate the dataset.

4. **Correlation-Based Feature Selection:** Computes a full Pearson correlation matrix across numerical features and filters for variables with |r| > 0.1 against `loan_status_number`, yielding a focused numerical feature set. Applies Point-Biserial correlation across all remaining categorical columns — identifying `grade` and `sub_grade` as the only categoricals with meaningful predictive signal — and appends them to the final feature list after label encoding via `cat.codes`.

5. **Decision Tree Classification:** Trains a `DecisionTreeClassifier` with `max_depth=10` on the selected feature set to produce an interpretable, rule-based credit risk model. Demonstrates end-to-end prediction on a sample borrower profile, returning a classification output that designates the applicant as Qualified or Unqualified.

6. **Random Forest Classification & Feature Importance:** Applies `StandardScaler` preprocessing to all numerical features before training a `RandomForestClassifier` with `max_depth=50` on an 80/20 stratified train-test split. Computes and ranks feature importances across all input variables, identifying `last_pymnt_d_month` as the most predictive feature — a finding that surfaces recency of payment behavior as the dominant signal in credit risk classification.

## 🛠️ Python Libraries & Techniques Reference

| Category | Library / Technique | Implementation |
|---|---|---|
| Data Manipulation | `pandas`, `numpy` | DataFrame operations, type casting, missing value handling, IQR outlier detection |
| Data Visualization | `plotly.express`, `seaborn`, `matplotlib` | Choropleth map, bar charts, boxplots, correlation heatmap |
| Statistical Analysis | `scipy.stats.pointbiserialr` | Correlation between binary target and categorical features |
| Feature Engineering | `pandas` label encoding, datetime decomposition | `cat.codes`, `str.replace`, month/year extraction |
| Preprocessing | `sklearn.preprocessing.StandardScaler` | Feature standardization prior to Random Forest training |
| Modeling | `sklearn.tree.DecisionTreeClassifier` | Interpretable rule-based binary classification (`max_depth=10`) |
| Modeling | `sklearn.ensemble.RandomForestClassifier` | Ensemble binary classification with feature importance (`max_depth=50`) |
| Model Evaluation | `sklearn.model_selection.train_test_split` | 80/20 stratified split (`random_state=42`) |

## 💡 Key Findings Summary

- **Recency of Payment Is the Strongest Predictor of Credit Risk:** `last_pymnt_d_month` ranks as the most important feature in the Random Forest model, establishing that *when* a borrower last made a payment is a more discriminating signal than how much they borrowed or their stated income. This implies that behavioral payment data should be weighted more heavily than static loan attributes in credit scoring models.
- **Loan Grade and Sub-Grade Are the Only Categorical Signals With Predictive Value:** Of all categorical features in the dataset, only `grade` and `sub_grade` clear the Point-Biserial correlation threshold, confirming that the lender's internal risk grading system meaningfully encodes default probability — and that other categorical attributes such as home ownership and employment length contribute marginal discriminatory power on their own.
- **The Dataset Is Structurally Imbalanced Toward Qualified Borrowers:** Binary encoding of `loan_status` reveals that Qualified borrowers (label = 1) significantly outnumber Unqualified borrowers (label = 0), with the "Current" status alone accounting for over 224,226 records. This class imbalance is a material consideration for model generalization, as a naïve classifier could achieve high accuracy by predicting the majority class without learning meaningful risk boundaries.
- **California Dominates Loan Application Volume:** Geographic analysis via choropleth mapping identifies California as the state with the highest concentration of loan applicants — a finding with direct implications for regional credit risk exposure and portfolio concentration management.
- **Feature Reduction From 74 to a Focused Predictive Set Is Validated Statistically:** The combination of correlation filtering (|r| > 0.1) and Point-Biserial analysis reduces the initial 74-column raw feature space to a compact, statistically justified set of predictors — demonstrating that rigorous feature selection improves model interpretability without sacrificing predictive coverage.

## 👤 About The Author

**Arruum Pratistha Kiranadjie**  
Data Analyst | Quantitative Analyst | Operations Research Analyst

Data-driven professional with a solid background in data analytics and quantitative research. Experienced in transforming complex datasets into actionable business insights using statistical methods, data visualization, and data modeling. Proven success leading end-to-end research projects across various industries. Passionate about leveraging data to drive strategic decisions and business growth.

- [LinkedIn: Arruum Kiranadjie](https://www.linkedin.com/in/arruumkiranadjie)
- [GitHub: Arruum Kiranadjie](https://github.com/arruumkiranadjie)
- [Tableau: Arruum Kiranadjie](https://public.tableau.com/app/profile/arruum.kiranadjie/vizzes)

© 2026 Arruum Kiranadjie. All rights reserved.
