import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import shap

df = pd.read_csv("/Users/ping/churn-bigml-20.csv")

# === 1. 把 Account length（天）转换成月份 cohort ===
df['tenure_months'] = (df['Account length'] / 30).astype(int)
df['cohort'] = pd.cut(
    df['tenure_months'],
    bins=[0, 1, 2, 3, 4, 5, 6, 99],
    labels=['0-1m', '1-2m', '2-3m', '3-4m', '4-5m', '5-6m', '6m+']
)

# === 2. Cohort Retention Table ===
# 按 cohort 和 customer service calls 分析 churn rate
cohort_stats = df.groupby('cohort').agg(
    total=('Churn', 'count'),
    churned=('Churn', 'sum')
).reset_index()
cohort_stats['retention_rate'] = 1 - (cohort_stats['churned'] / cohort_stats['total'])
cohort_stats['churn_rate'] = cohort_stats['churned'] / cohort_stats['total']

print("=== Cohort Retention by Tenure ===")
print(cohort_stats.to_string(index=False))

# === 3. 画 Cohort Heatmap（类似你截图的样式）===
# 用 plan type 和 tenure 做 2D cohort
df['plan_type'] = df.apply(lambda x:
    'Intl+VM' if x['International plan'] == 'Yes' and x['Voice mail plan'] == 'Yes'
    else 'Intl only' if x['International plan'] == 'Yes'
    else 'VM only' if x['Voice mail plan'] == 'Yes'
    else 'Basic', axis=1
)

pivot = df.pivot_table(
    values='Churn',
    index='plan_type',
    columns='cohort',
    aggfunc=lambda x: 1 - x.mean()  # retention rate
).round(2)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle("Subscription Retention Analysis\n(Telecom Churn Dataset — proxy for music streaming)",
             fontsize=14, fontweight='bold')

# Heatmap
sns.heatmap(
    pivot,
    ax=axes[0],
    annot=True,
    fmt='.2f',
    cmap='Blues',
    vmin=0.5, vmax=1.0,
    linewidths=0.5,
    cbar_kws={'label': 'Retention Rate'}
)
axes[0].set_title("Retention Rate by Plan Type & Tenure Cohort", fontsize=12)
axes[0].set_xlabel("Tenure Cohort")
axes[0].set_ylabel("Subscription Plan Type")

# Retention curve by cohort
for cohort_label in cohort_stats['cohort'].unique():
    subset = df[df['cohort'] == cohort_label]
    # customer service calls 作为 engagement proxy
    cs_bins = pd.cut(subset['Customer service calls'],
                     bins=[-1, 0, 1, 2, 3, 10],
                     labels=['0', '1', '2', '3', '4+'])
    retention_by_cs = subset.groupby(cs_bins, observed=True)['Churn'].apply(
        lambda x: 1 - x.mean()
    )

axes[1].set_title("Churn Rate by Number of Support Contacts\n(proxy: users reaching out = at-risk signal)",
                  fontsize=12)

churn_by_cs = df.groupby(
    pd.cut(df['Customer service calls'], bins=[-1,0,1,2,3,10],
           labels=['0','1','2','3','4+'])
)['Churn'].mean()

bars = axes[1].bar(churn_by_cs.index, churn_by_cs.values,
                   color=['#1DB954' if v < 0.2 else '#FFA500' if v < 0.4 else '#FF4444'
                          for v in churn_by_cs.values])
axes[1].set_xlabel("Number of Customer Service Contacts")
axes[1].set_ylabel("Churn Rate")
axes[1].axhline(y=df['Churn'].mean(), color='gray', linestyle='--',
                label=f'Overall avg: {df["Churn"].mean():.1%}')
axes[1].legend()
for bar, val in zip(bars, churn_by_cs.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("/Users/ping/churn_cohort.png", dpi=150, bbox_inches='tight')
print("\n✅ 图表已保存到 /Users/ping/churn_cohort.png")
plt.show()


# === 4. Churn 预测模型 ===
df_model = df.copy()

# 编码 categorical 字段
for col in ['State', 'International plan', 'Voice mail plan']:
    df_model[col] = LabelEncoder().fit_transform(df_model[col])

# 去掉 cohort 和 plan_type（派生字段）
features = [c for c in df_model.columns
            if c not in ['Churn', 'cohort', 'plan_type', 'tenure_months']]
X = df_model[features]
y = df_model['Churn'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== 模型性能 ===")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
print(classification_report(y_test, y_pred))

# === 5. SHAP 可解释性 ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Churn Prediction Model — Feature Importance (SHAP)",
             fontsize=14, fontweight='bold')

# SHAP summary plot
plt.sca(axes[0])
shap.summary_plot(shap_values, X_test, plot_type="bar",
                  show=False, max_display=10)
axes[0].set_title("Top 10 Churn Drivers")

# Feature importance bar chart（备用，更清晰）
importances = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X_test.columns
).sort_values(ascending=True).tail(10)

axes[1].barh(importances.index, importances.values, color='#1DB954')
axes[1].set_title("Mean |SHAP Value| — Impact on Churn")
axes[1].set_xlabel("Mean |SHAP Value|")

plt.tight_layout()
plt.savefig("/Users/ping/churn_model.png", dpi=150, bbox_inches='tight')
print("\n✅ SHAP 图表已保存到 /Users/ping/churn_model.png")
plt.show()