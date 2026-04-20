# Subscription Churn & Retention Analysis

**Live dashboard:** [subscription-churn-analysis.streamlit.app](https://subscription-churn-analysis.streamlit.app)

An end-to-end data science project analysing subscriber behaviour to identify churn drivers, model at-risk users, and recommend targeted retention interventions — directly analogous to the analytical work done on a music streaming subscription team.

---

## Business Questions

- Which subscriber segments have the lowest retention rates, and why?
- What behavioural signals predict churn before it happens?
- How should at-risk subscribers be prioritised for intervention?

---

## Key Findings

**Retention varies significantly by plan type and tenure:**

VM-only subscribers retain at 91–98% across all tenure cohorts, suggesting that feature adoption is the strongest driver of loyalty. Intl+VM subscribers churn at the highest rate (44–67%) despite holding the highest-value plans — indicating that high-price plans require active value reinforcement to retain. Subscribers who survive past 6 months almost never churn, pointing to a loyalty threshold that interventions should aim to help users cross.

**Support contact volume is the strongest churn signal:**

Subscribers with 4 or more support contacts churn at 47.4%, compared to a 14.2% overall average — a 3x uplift. This is a clear, actionable early-warning signal that can be used to trigger proactive outreach before a subscriber cancels.

**Churn prediction model:**

A Gradient Boosting classifier achieves AUC-ROC 0.916 on held-out data, well above the 0.9 threshold generally considered production-ready. SHAP analysis confirms that support contact volume, daily usage intensity, and plan type are the top three churn drivers.

---

## Dashboard

The interactive dashboard has three sections:

**Cohort Retention** — retention heatmap by plan type and tenure cohort, triangular cohort retention table, retained vs churned subscriber counts by support contact level, and subscriber volume vs churn rate by tenure.

**Churn Prediction** — SHAP-based feature importance chart with a mapping of model features to their streaming subscription equivalents.

**At-Risk Users** — adjustable probability threshold to identify and prioritise at-risk subscribers, with a ranked list and recommended intervention strategies by risk tier.

---

## Methodology

| Step | Description |
|---|---|
| Data preparation | Tenure bucketing into monthly cohorts, plan type engineering (Basic / VM only / Intl only / Intl+VM), cohort assignment |
| Cohort retention analysis | Retention rate by plan type × tenure cohort heatmap; triangular cohort survival table |
| Behavioural analysis | Churn rate and subscriber counts segmented by support contact frequency |
| Predictive modelling | Gradient Boosting Classifier with stratified 80/20 train/test split |
| Explainability | SHAP TreeExplainer for feature-level attribution across the full dataset |
| Deployment | Streamlit Cloud for public interactive access |

---

## Dataset

**Source:** UCI Telecom Churn Dataset via [Kaggle](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets)

667 subscriber records, 20 features, zero missing values. Overall churn rate of 14.2% reflects a realistic class imbalance rather than a synthetic 50/50 split.

Used as a proxy for subscription business dynamics — the underlying behavioural patterns (tenure, plan type, usage intensity, support contact volume, churn) are structurally analogous to music streaming subscription models.

---

## Repository Structure

```
subscription-churn-analysis/
├── churn-bigml-20.csv      # Dataset
├── churn_analysis.py       # Exploratory analysis and static charts
├── dashboard.py            # Streamlit dashboard
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/pingyan-data/subscription-churn-analysis.git
cd subscription-churn-analysis
pip install -r requirements.txt
streamlit run dashboard.py
```

---

## Tech Stack

Python · Pandas · Scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn · Streamlit

---

## Author

Ping Yan — Senior Data Scientist, Stockholm
[LinkedIn](https://www.linkedin.com/in/ping-yan/) · [GitHub](https://github.com/pingyan-data)
