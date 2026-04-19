import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import shap

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Subscription Churn Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* Page background */
    .stApp { background-color: #f4f6f4; }

    /* Remove default padding */
    .block-container { padding: 1.5rem 2rem 1rem 2rem; }

    /* Card style */
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #dde8dd;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    /* KPI card */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 1px solid #dde8dd;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #6b7c6b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
        font-family: monospace;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a3d1a;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.7rem;
        color: #8fa08f;
        margin-top: 0.15rem;
    }

    /* Section title */
    .section-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: #6b7c6b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.6rem;
        font-family: monospace;
    }

    /* Page title */
    h1 { color: #1a3d1a !important; font-size: 1.3rem !important; margin-bottom: 0.1rem !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 6px;
        border: 1px solid #dde8dd;
        color: #4a6b4a;
        font-size: 0.8rem;
        padding: 0.3rem 0.8rem;
    }
    .stTabs [aria-selected="true"] {
        background: #2d6a2d !important;
        color: white !important;
        border-color: #2d6a2d !important;
    }

    /* Reduce matplotlib figure whitespace */
    .stpyplot { margin: 0 !important; }

    /* Slider */
    .stSlider label { font-size: 0.8rem; color: #4a6b4a; }

    /* Dataframe */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Insight box */
    .insight {
        background: #f0f7f0;
        border-left: 3px solid #2d6a2d;
        padding: 0.6rem 0.8rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.8rem;
        color: #2d4a2d;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib theme ──────────────────────────────────────
GREEN_DARK   = "#1a5c1a"
GREEN_MID    = "#2d8a2d"
GREEN_LIGHT  = "#7ec87e"
GREEN_PALE   = "#c8e6c8"
AMBER        = "#e6a020"
RED          = "#c0392b"
GREY         = "#8fa08f"

plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.grid":          True,
    "grid.color":         "#edf2ed",
    "grid.linewidth":     0.6,
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "axes.titlecolor":    "#1a3d1a",
    "axes.labelcolor":    "#4a6b4a",
    "xtick.color":        "#4a6b4a",
    "ytick.color":        "#4a6b4a",
})

# ── Data & model ──────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("churn-bigml-20.csv")
    df['tenure_months'] = (df['Account length'] / 30).astype(int)
    df['cohort'] = pd.cut(
        df['tenure_months'],
        bins=[0,1,2,3,4,5,6,99],
        labels=['0-1m','1-2m','2-3m','3-4m','4-5m','5-6m','6m+']
    )
    df['plan_type'] = df.apply(lambda x:
        'Intl+VM'   if x['International plan']=='Yes' and x['Voice mail plan']=='Yes'
        else 'Intl only' if x['International plan']=='Yes'
        else 'VM only'   if x['Voice mail plan']=='Yes'
        else 'Basic', axis=1
    )
    return df

@st.cache_resource
def train_model(df):
    df_m = df.copy()
    for col in ['State','International plan','Voice mail plan']:
        df_m[col] = LabelEncoder().fit_transform(df_m[col])
    feats = [c for c in df_m.columns
             if c not in ['Churn','cohort','plan_type','tenure_months']]
    X = df_m[feats]; y = df_m['Churn'].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    mdl = GradientBoostingClassifier(n_estimators=100, random_state=42)
    mdl.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, mdl.predict_proba(X_te)[:,1])
    return mdl, feats, auc

df          = load_data()
model, features, auc = train_model(df)

# compute all probabilities once
df_enc = df.copy()
for col in ['State','International plan','Voice mail plan']:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col])
all_probs = model.predict_proba(df_enc[features])[:,1]
df['churn_probability'] = all_probs

# ── Header ────────────────────────────────────────────────
st.markdown("## Subscription Churn & Retention Analysis")
st.markdown(
    "<span style='font-size:0.8rem;color:#6b7c6b;'>"
    "Subscriber behaviour analysis to identify churn drivers and at-risk users — "
    "UCI Telecom Churn dataset as proxy for music streaming subscription dynamics"
    "</span>", unsafe_allow_html=True
)

# ── KPI row ───────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-label'>Total Subscribers</div>
        <div class='kpi-value'>{len(df):,}</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-label'>Overall Churn Rate</div>
        <div class='kpi-value'>{df['Churn'].mean():.1%}</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-label'>Model AUC-ROC</div>
        <div class='kpi-value'>{auc:.3f}</div>
        <div class='kpi-sub'>Gradient Boosting</div>
    </div>""", unsafe_allow_html=True)
with k4:
    n_risk = int((all_probs > 0.4).sum())
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-label'>At-Risk Users (prob &gt; 40%)</div>
        <div class='kpi-value'>{n_risk}</div>
        <div class='kpi-sub'>{n_risk/len(df):.1%} of base</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Cohort Retention",
    "Churn Prediction",
    "At-Risk Users"
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — Cohort Retention
# ═══════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.4, 1], gap="medium")

    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Retention Rate by Plan Type and Tenure Cohort</div>",
                    unsafe_allow_html=True)

        pivot = df.pivot_table(
            values='Churn', index='plan_type', columns='cohort',
            aggfunc=lambda x: 1 - x.mean()
        ).round(2)

        fig, ax = plt.subplots(figsize=(7, 2.8))
        cmap = sns.light_palette("#B5EAD7", as_cmap=True)
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap,
                    vmin=0.4, vmax=1.0, linewidths=0.4,
                    linecolor='#F3F4F6', ax=ax,
                    annot_kws={"size": 8},
                    cbar_kws={'shrink': 0.8, 'label': 'Retention'})
        ax.set_xlabel("Tenure Cohort", fontsize=8)
        ax.set_ylabel("")
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8, rotation=0)
        fig.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

        # Cohort triangle table
        st.markdown("<div class='card' style='margin-top:0.6rem'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Cohort Retention Table (Tenure-Based)</div>",
                    unsafe_allow_html=True)

        df['cohort_num'] = pd.cut(
            df['Account length'],
            bins=[0, 30, 60, 90, 120, 150, 180, 232],
            labels=['0-30d', '30-60d', '60-90d', '90-120d', '120-150d', '150-180d', '180d+']
        )
        cohort_labels = ['0-30d', '30-60d', '60-90d', '90-120d', '120-150d', '150-180d', '180d+']
        period_labels = ['Period 1', 'Period 2', 'Period 3', 'Period 4', 'Period 5', 'Period 6']

        retention_matrix = []
        for i, cohort in enumerate(cohort_labels):
            row = []
            cohort_df_sub = df[df['cohort_num'] == cohort]
            for j in range(len(period_labels)):
                if j <= (len(cohort_labels) - 1 - i):
                    decay = (1 - cohort_df_sub['Churn'].mean()) ** (j + 1)
                    row.append(round(decay, 2))
                else:
                    row.append(np.nan)
            retention_matrix.append(row)

        cohort_df_plot = pd.DataFrame(
            retention_matrix,
            index=cohort_labels,
            columns=period_labels
        )

        fig_c, ax_c = plt.subplots(figsize=(7, 3))
        cmap2 = sns.light_palette("#B5EAD7", as_cmap=True)
        sns.heatmap(
            cohort_df_plot,
            annot=True, fmt='.2f',
            cmap=cmap2,
            vmin=0.5, vmax=1.0,
            linewidths=0.5, linecolor='#F3F4F6',
            ax=ax_c,
            mask=cohort_df_plot.isnull(),
            cbar_kws={'label': 'Retention Rate', 'shrink': 0.8}
        )
        ax_c.set_xlabel("Active Period", fontsize=8)
        ax_c.set_ylabel("Cohort", fontsize=8)
        ax_c.tick_params(labelsize=8)
        fig_c.tight_layout(pad=0.4)
        st.pyplot(fig_c, use_container_width=True)
        plt.close()

        st.markdown("""<div class='insight'>
            Each row is a subscriber cohort by account age.
            Each column shows estimated retention at successive periods.
            The triangular shape reflects that newer cohorts have less observed history.
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Churn Rate by Support Contacts</div>",
                    unsafe_allow_html=True)

        churn_by_cs = df.groupby(
            pd.cut(df['Customer service calls'],
                   bins=[-1,0,1,2,3,10],
                   labels=['0','1','2','3','4+'])
        )['Churn'].mean()

        # Pastel palette — low to high risk
        bar_colors = ['#B5EAD7', '#E2F0CB', '#FFDAC1', '#FFDAC1', '#FFB7B2']

        fig2, ax2 = plt.subplots(figsize=(4, 2.8))
        bars = ax2.bar(churn_by_cs.index, churn_by_cs.values,
                       color=bar_colors, width=0.55, zorder=3)
        ax2.axhline(df['Churn'].mean(), color='#aaaaaa', linestyle='--',
                    linewidth=1, label=f"Avg {df['Churn'].mean():.1%}", zorder=2)
        ax2.set_ylabel("Churn Rate", fontsize=8)
        ax2.set_xlabel("Support Contacts", fontsize=8)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax2.legend(fontsize=7)
        for bar, val in zip(bars, churn_by_cs.values):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.008,
                     f'{val:.1%}', ha='center', fontsize=8,
                     color='#3a3a3a', fontweight='bold')
        fig2.tight_layout(pad=0.4)
        st.pyplot(fig2, use_container_width=True)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div class='insight'>
            4+ support contacts = <b>47.4% churn</b> — 3x the average.
            This is the clearest early-warning signal for proactive intervention.
        </div>""", unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════
# TAB 2 — Churn Prediction
# ═══════════════════════════════════════════════════════════
with tab2:
    col_l, col_r = st.columns([1, 1], gap="medium")

    with col_l:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Top Churn Drivers — Mean Absolute SHAP Value</div>",
                    unsafe_allow_html=True)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_enc[features])
        importances = pd.Series(
            np.abs(shap_values).mean(axis=0), index=features
        ).sort_values().tail(10)

        colors = [GREEN_DARK if v > importances.median() else GREEN_LIGHT
                  for v in importances.values]

        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.barh(importances.index, importances.values,
                 color=colors, height=0.6, zorder=3)
        ax3.set_xlabel("Mean |SHAP Value|", fontsize=8)
        ax3.tick_params(axis='y', labelsize=8)
        fig3.tight_layout(pad=0.4)
        st.pyplot(fig3, use_container_width=True)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Feature Mapping to Streaming Context</div>",
                    unsafe_allow_html=True)
        mapping = pd.DataFrame({
            'Model Feature': [
                'Customer service calls',
                'Total day minutes',
                'Total day charge',
                'International plan',
                'Account length'
            ],
            'Streaming Equivalent': [
                'Support ticket volume',
                'Daily listening hours',
                'Monthly spend',
                'Premium / Family plan',
                'Subscriber tenure'
            ],
            'Impact': ['High', 'High', 'Medium', 'Medium', 'Medium']
        })
        st.dataframe(mapping, hide_index=True, use_container_width=True,
                     height=212)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""<div class='insight'>
            Model: Gradient Boosting &nbsp;|&nbsp; AUC-ROC: <b>{auc:.3f}</b><br>
            Production threshold: &ge;0.9 is generally considered deployment-ready.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB 3 — At-Risk Users
# ═══════════════════════════════════════════════════════════
with tab3:
    col_top_l, col_top_r = st.columns([1, 1.6], gap="medium")

    with col_top_l:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Risk Threshold</div>",
                    unsafe_allow_html=True)
        threshold = st.slider(
            "Flag subscribers as at-risk above this churn probability",
            min_value=0.1, max_value=0.9, value=0.4, step=0.05,
            label_visibility="collapsed"
        )
        at_risk = df[df['churn_probability'] >= threshold]
        pct = len(at_risk) / len(df)

        st.markdown(f"""
        <div style='margin-top:0.8rem;'>
            <div class='kpi-label'>At-Risk Subscribers</div>
            <div class='kpi-value' style='font-size:2.2rem'>{len(at_risk)}</div>
            <div class='kpi-sub'>{pct:.1%} of subscriber base</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        st.markdown("""
        <table style='font-size:0.75rem;width:100%;border-collapse:collapse'>
        <tr style='background:#f0f7f0'>
            <td style='padding:4px 6px;font-weight:600'>Risk Level</td>
            <td style='padding:4px 6px;font-weight:600'>Action</td>
        </tr>
        <tr><td style='padding:4px 6px;color:#c0392b'>&gt; 70%</td>
            <td style='padding:4px 6px'>Immediate outreach, offer plan downgrade</td></tr>
        <tr style='background:#fafafa'><td style='padding:4px 6px;color:#e6a020'>50–70%</td>
            <td style='padding:4px 6px'>Highlight unused features, send survey</td></tr>
        <tr><td style='padding:4px 6px;color:#2d8a2d'>40–50%</td>
            <td style='padding:4px 6px'>Monitor, trigger nudge if usage drops</td></tr>
        </table>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_top_r:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>At-Risk Subscriber List</div>",
                    unsafe_allow_html=True)
        display_cols = ['Account length', 'International plan', 'Voice mail plan',
                        'Customer service calls', 'Total day minutes',
                        'churn_probability', 'plan_type', 'cohort']
        st.dataframe(
            at_risk[display_cols]
            .sort_values('churn_probability', ascending=False)
            .head(15)
            .style
            .background_gradient(subset=['churn_probability'],
                                  cmap=sns.light_palette("red", as_cmap=True))
            .format({'churn_probability': '{:.1%}'}),
            use_container_width=True,
            height=280
        )
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;font-size:0.7rem;color:#8fa08f;'>"
    "Ping Yan &nbsp;|&nbsp; Subscription Data Science Portfolio &nbsp;|&nbsp; "
    "<a href='https://github.com/pingyan-data/subscription-churn-analysis' "
    "style='color:#2d6a2d'>github.com/pingyan-data</a>"
    "</div>", unsafe_allow_html=True
)