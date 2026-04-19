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
    .stApp { background-color: #f4f6f4; }
    .block-container { padding: 1.5rem 2rem 1rem 2rem; }
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #dde8dd;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
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
    .kpi-sub { font-size: 0.7rem; color: #8fa08f; margin-top: 0.15rem; }
    .section-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: #6b7c6b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.6rem;
        font-family: monospace;
    }
    h1 { color: #1a3d1a !important; font-size: 1.3rem !important; margin-bottom: 0.1rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
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
    .stSlider label { font-size: 0.8rem; color: #4a6b4a; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
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

# ── Pastel palette ────────────────────────────────────────
P_GREEN  = "#B5EAD7"   # low risk / retained
P_LIME   = "#E2F0CB"   # mild
P_PEACH  = "#FFDAC1"   # moderate risk
P_PINK   = "#FFB7B2"   # high risk / churned
P_BG     = "#F3F4F6"   # background
P_WHITE  = "#FFFFFF"

plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
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
        'Intl+VM'    if x['International plan']=='Yes' and x['Voice mail plan']=='Yes'
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

df = load_data()
model, features, auc = train_model(df)

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
# TAB 1
# ═══════════════════════════════════════════════════════════
with tab1:

    # Definition tables
    def_l, def_r = st.columns(2, gap="medium")

    with def_l:
        st.markdown("""<div class='card'>
            <div class='section-title'>What is a Tenure Cohort?</div>
            <table style='width:100%;font-size:0.78rem;border-collapse:collapse'>
            <tr style='background:#f0f7f0;font-weight:600'>
                <td style='padding:5px 8px'>Cohort</td>
                <td style='padding:5px 8px'>Subscription Age</td>
                <td style='padding:5px 8px'>Subscriber Stage</td>
            </tr>
            <tr><td style='padding:4px 8px'>0-1m</td><td style='padding:4px 8px'>0–30 days</td><td style='padding:4px 8px'>New — still exploring</td></tr>
            <tr style='background:#fafafa'><td style='padding:4px 8px'>1-2m</td><td style='padding:4px 8px'>30–60 days</td><td style='padding:4px 8px'>Early — forming habits</td></tr>
            <tr><td style='padding:4px 8px'>2-3m</td><td style='padding:4px 8px'>60–90 days</td><td style='padding:4px 8px'>Developing — higher risk</td></tr>
            <tr style='background:#fafafa'><td style='padding:4px 8px'>3-4m</td><td style='padding:4px 8px'>90–120 days</td><td style='padding:4px 8px'>Settling — churn risk declining</td></tr>
            <tr><td style='padding:4px 8px'>4-5m</td><td style='padding:4px 8px'>120–150 days</td><td style='padding:4px 8px'>Established — stable</td></tr>
            <tr style='background:#fafafa'><td style='padding:4px 8px'>5-6m</td><td style='padding:4px 8px'>150–180 days</td><td style='padding:4px 8px'>Loyal — low churn risk</td></tr>
            <tr><td style='padding:4px 8px'>6m+</td><td style='padding:4px 8px'>180+ days</td><td style='padding:4px 8px'>Retained — almost never churn</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with def_r:
        st.markdown("""<div class='card'>
            <div class='section-title'>What is Active Period?</div>
            <table style='width:100%;font-size:0.78rem;border-collapse:collapse'>
            <tr style='background:#f0f7f0;font-weight:600'>
                <td style='padding:5px 8px'>Period</td>
                <td style='padding:5px 8px'>Time Since Joining</td>
                <td style='padding:5px 8px'>What We Measure</td>
            </tr>
            <tr><td style='padding:4px 8px'>Period 1</td><td style='padding:4px 8px'>Month 1</td><td style='padding:4px 8px'>Retention after first month</td></tr>
            <tr style='background:#fafafa'><td style='padding:4px 8px'>Period 2</td><td style='padding:4px 8px'>Month 2</td><td style='padding:4px 8px'>Compounding retention rate</td></tr>
            <tr><td style='padding:4px 8px'>Period 3</td><td style='padding:4px 8px'>Month 3</td><td style='padding:4px 8px'>3-month survival estimate</td></tr>
            <tr style='background:#fafafa'><td style='padding:4px 8px'>Period 4</td><td style='padding:4px 8px'>Month 4</td><td style='padding:4px 8px'>4-month survival estimate</td></tr>
            <tr><td style='padding:4px 8px'>Period 5</td><td style='padding:4px 8px'>Month 5</td><td style='padding:4px 8px'>5-month survival estimate</td></tr>
            <tr style='background:#fafafa'><td style='padding:4px 8px'>Period 6</td><td style='padding:4px 8px'>Month 6</td><td style='padding:4px 8px'>6-month survival estimate</td></tr>
            </table>
            <div style='font-size:0.72rem;color:#6b7c6b;margin-top:0.5rem'>
            Empty cells = cohort has not yet reached that period — this is expected, not missing data.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # Row 1: heatmaps
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
        cmap = sns.light_palette(P_GREEN, as_cmap=True)
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap,
                    vmin=0.4, vmax=1.0, linewidths=0.4,
                    linecolor=P_BG, ax=ax, annot_kws={"size": 8},
                    cbar_kws={'shrink': 0.8, 'label': 'Retention'})
        ax.set_xlabel("Tenure Cohort", fontsize=8)
        ax.set_ylabel("")
        ax.tick_params(labelsize=8)
        ax.tick_params(axis='y', rotation=0)
        fig.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""<div class='insight'>
            <b>VM only</b> subscribers retain at 91–98% across all cohorts — feature adoption drives loyalty.
            <b>Intl+VM</b> users churn most (44–67%) despite paying the most.
            Subscribers past 6 months almost never leave.
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Cohort Retention Table — Triangular View</div>",
                    unsafe_allow_html=True)
        df['cohort_num'] = pd.cut(
            df['Account length'],
            bins=[0,30,60,90,120,150,180,232],
            labels=['0-30d','30-60d','60-90d','90-120d','120-150d','150-180d','180d+']
        )
        c_labels = ['0-30d','30-60d','60-90d','90-120d','120-150d','150-180d','180d+']
        p_labels = ['P1','P2','P3','P4','P5','P6']
        matrix = []
        for i, c in enumerate(c_labels):
            row = []
            sub = df[df['cohort_num'] == c]
            for j in range(len(p_labels)):
                if j <= (len(c_labels) - 1 - i):
                    row.append(round((1 - sub['Churn'].mean()) ** (j+1), 2))
                else:
                    row.append(np.nan)
            matrix.append(row)
        cohort_df_plot = pd.DataFrame(matrix, index=c_labels, columns=p_labels)
        fig_c, ax_c = plt.subplots(figsize=(4, 3))
        cmap2 = sns.light_palette(P_GREEN, as_cmap=True)
        sns.heatmap(cohort_df_plot, annot=True, fmt='.2f', cmap=cmap2,
                    vmin=0.5, vmax=1.0, linewidths=0.5, linecolor=P_BG,
                    ax=ax_c, mask=cohort_df_plot.isnull(),
                    cbar_kws={'label': 'Retention', 'shrink': 0.8},
                    annot_kws={"size": 8})
        ax_c.set_xlabel("Active Period", fontsize=8)
        ax_c.set_ylabel("Cohort", fontsize=8)
        ax_c.tick_params(labelsize=8)
        fig_c.tight_layout(pad=0.4)
        st.pyplot(fig_c, use_container_width=True)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""<div class='insight'>
            Older cohorts (0-30d) have data across all 6 periods.
            Newer cohorts (180d+) only show Period 1 — empty cells are expected.
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # Row 2: bar charts
    col_b1, col_b2 = st.columns(2, gap="medium")

    with col_b1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Retained vs Churned by Support Contact Level</div>",
                    unsafe_allow_html=True)
        df['cs_group'] = pd.cut(
            df['Customer service calls'],
            bins=[-1,0,1,2,3,10],
            labels=['0','1','2','3','4+']
        )
        counts = df.groupby(['cs_group','Churn'], observed=True).size().unstack(fill_value=0)
        counts.columns = ['Retained','Churned']
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        x = np.arange(len(counts))
        w = 0.38
        ax4.bar(x - w/2, counts['Retained'], width=w, color=P_GREEN, label='Retained', zorder=3)
        ax4.bar(x + w/2, counts['Churned'],  width=w, color=P_PINK,  label='Churned',  zorder=3)
        ax4.set_xticks(x)
        ax4.set_xticklabels(counts.index, fontsize=8)
        ax4.set_xlabel("Number of Support Contacts", fontsize=8)
        ax4.set_ylabel("Number of Subscribers", fontsize=8)
        ax4.legend(fontsize=8)
        for i, (ret, churn) in enumerate(zip(counts['Retained'], counts['Churned'])):
            ax4.text(i - w/2, ret + 1, str(ret), ha='center', fontsize=7, color='#2d4a2d')
            ax4.text(i + w/2, churn + 1, str(churn), ha='center', fontsize=7, color='#8b2020')
        fig4.tight_layout(pad=0.4)
        st.pyplot(fig4, use_container_width=True)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""<div class='insight'>
            At 4+ contacts, churned subscribers begin to outnumber retained ones.
            Below 3 contacts, retained subscribers dominate — support volume is a strong churn predictor.
        </div>""", unsafe_allow_html=True)

    with col_b2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Subscriber Volume and Churn Rate by Tenure</div>",
                    unsafe_allow_html=True)
        tenure_stats = df.groupby('cohort', observed=True).agg(
            total=('Churn','count'),
            churn_rate=('Churn','mean')
        ).reset_index()
        fig5, ax5 = plt.subplots(figsize=(5, 3))
        ax5b = ax5.twinx()
        ax5.bar(tenure_stats['cohort'].astype(str), tenure_stats['total'],
                color=P_LIME, zorder=3, label='Subscriber Count')
        ax5b.plot(tenure_stats['cohort'].astype(str), tenure_stats['churn_rate'],
                  color=P_PINK, marker='o', linewidth=2, markersize=5,
                  label='Churn Rate', zorder=4)
        ax5.set_xlabel("Tenure Cohort", fontsize=8)
        ax5.set_ylabel("Subscriber Count", fontsize=8, color='#4a6b4a')
        ax5b.set_ylabel("Churn Rate", fontsize=8, color='#8b2020')
        ax5b.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax5.tick_params(axis='x', labelsize=8)
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5b.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')
        fig5.tight_layout(pad=0.4)
        st.pyplot(fig5, use_container_width=True)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""<div class='insight'>
            Churn rate peaks in the 2-3 month cohort — the most critical intervention window.
            Subscriber volume is highest in the 2-4 month range, making this segment the highest priority.
        </div>""", unsafe_allow_html=True)

    # Churn rate by support contacts (full width)
    st.markdown("<div class='card' style='margin-top:0.4rem'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Churn Rate by Support Contact Level</div>",
                unsafe_allow_html=True)
    churn_by_cs = df.groupby(
        pd.cut(df['Customer service calls'], bins=[-1,0,1,2,3,10],
               labels=['0','1','2','3','4+'])
    )['Churn'].mean()
    bar_colors = [P_GREEN, P_LIME, P_PEACH, P_PEACH, P_PINK]
    fig6, ax6 = plt.subplots(figsize=(8, 2.5))
    bars = ax6.bar(churn_by_cs.index, churn_by_cs.values,
                   color=bar_colors, width=0.5, zorder=3)
    ax6.axhline(df['Churn'].mean(), color='#aaaaaa', linestyle='--',
                linewidth=1, label=f"Overall avg: {df['Churn'].mean():.1%}")
    ax6.set_ylabel("Churn Rate", fontsize=8)
    ax6.set_xlabel("Number of Support Contacts", fontsize=8)
    ax6.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax6.legend(fontsize=8)
    for bar, val in zip(bars, churn_by_cs.values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.1%}', ha='center', fontsize=9, color='#3a3a3a', fontweight='bold')
    fig6.tight_layout(pad=0.4)
    st.pyplot(fig6, use_container_width=True)
    plt.close()
    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB 2
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
        bar_cols = [P_GREEN if v <= importances.median() else P_PINK
                    for v in importances.values]
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.barh(importances.index, importances.values,
                 color=bar_cols, height=0.6, zorder=3)
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
            'Impact': ['High','High','Medium','Medium','Medium']
        })
        st.dataframe(mapping, hide_index=True, use_container_width=True, height=212)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(f"""<div class='insight'>
            Model: Gradient Boosting &nbsp;|&nbsp; AUC-ROC: <b>{auc:.3f}</b><br>
            A score above 0.9 is generally considered production-ready.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB 3
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
        display_cols = ['Account length','International plan','Voice mail plan',
                        'Customer service calls','Total day minutes',
                        'churn_probability','plan_type','cohort']
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