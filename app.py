"""
Purchase Probability Prediction — Streamlit Dashboard
Framework: SCQR Full Structure + Pyramid Principle per Page

Author: Reinaldi Santoso
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os, joblib

# ══════════════════════════════════════════════════════════════
# CONFIG & DATA LOADING
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Purchase Prediction — Reinaldi Santoso",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(__file__)

@st.cache_data
def load_config():
    with open(os.path.join(BASE, "model", "model_config.json")) as f:
        return json.load(f)

@st.cache_data
def load_comparison():
    return pd.read_csv(os.path.join(BASE, "data", "model_comparison_final.csv"), index_col=0)

@st.cache_data
def load_features():
    return pd.read_csv(os.path.join(BASE, "data", "feature_business_interpretation.csv"))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "model", "catboost_tuned.pkl"))

cfg = load_config()
df_models = load_comparison()
df_features = load_features()

# ══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp { font-family: 'DM Sans', sans-serif; }
    code, .stCode { font-family: 'JetBrains Mono', monospace; }

    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); border-color: #2563EB; }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #60A5FA; line-height: 1.2; }
    .metric-label { font-size: 0.85rem; color: #94A3B8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-sub { font-size: 0.75rem; color: #64748B; margin-top: 2px; }

    .scqr-box {
        border-left: 4px solid;
        padding: 16px 20px;
        margin: 12px 0;
        border-radius: 0 8px 8px 0;
        background: rgba(30, 41, 59, 0.5);
    }
    .scqr-s { border-color: #6366F1; }
    .scqr-c { border-color: #F59E0B; }
    .scqr-q { border-color: #EF4444; }
    .scqr-r { border-color: #10B981; }

    .scqr-tag {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 6px;
    }
    .tag-s { background: #6366F1; color: white; }
    .tag-c { background: #F59E0B; color: #1E293B; }
    .tag-q { background: #EF4444; color: white; }
    .tag-r { background: #10B981; color: white; }

    .pyramid-l1 {
        background: linear-gradient(135deg, #1E3A5F 0%, #1E293B 100%);
        border: 1px solid #2563EB;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
    }
    .pyramid-l2 {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }

    .insight-box {
        background: linear-gradient(135deg, #1a2332 0%, #162032 100%);
        border-left: 3px solid #60A5FA;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    }

    .footer-text {
        text-align: center;
        color: #475569;
        font-size: 0.75rem;
        margin-top: 60px;
        padding: 20px;
        border-top: 1px solid #1E293B;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════
def metric_card(value, label, sub=""):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)

def scqr_box(tag, title, content):
    colors = {"S": ("scqr-s", "tag-s"), "C": ("scqr-c", "tag-c"),
              "Q": ("scqr-q", "tag-q"), "R": ("scqr-r", "tag-r")}
    box_cls, tag_cls = colors.get(tag, ("scqr-s", "tag-s"))
    labels = {"S": "SITUATION", "C": "COMPLICATION", "Q": "QUESTION", "R": "RESOLUTION"}
    st.markdown(f"""
    <div class="scqr-box {box_cls}">
        <span class="scqr-tag {tag_cls}">{labels.get(tag, tag)}</span>
        <div style="font-weight:600; font-size:1.05rem; margin-bottom:6px;">{title}</div>
        <div style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">{content}</div>
    </div>""", unsafe_allow_html=True)

def footer():
    st.markdown("""
    <div class="footer-text">
        Portfolio by <strong>Reinaldi Santoso</strong> · Mining Engineering (ITB) · diBimbing Data Science Bootcamp<br>
        Dataset: eCommerce Events History in Electronics Store (Kaggle — mkechinov)
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛒 Purchase Prediction")
    st.markdown("##### Reinaldi Santoso")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Executive Summary",
         "📊 Data & Funnel",
         "🏆 Model Arena",
         "🔍 Why It Works",
         "🎯 Predict Now"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#64748B; line-height:1.5;">
    <strong>Framework:</strong><br>
    SCQR + Pyramid Principle<br><br>
    <strong>Best Model:</strong><br>
    CatBoost (Tuned)<br>
    AUC-ROC: 0.9953<br><br>
    <strong>Dataset:</strong><br>
    885K events · 490K sessions
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE SUMMARY (SCQR Full → Level 1 Pyramid)
# ══════════════════════════════════════════════════════════════
if page == "🏠 Executive Summary":

    st.markdown("# Purchase Probability Prediction")
    st.markdown("##### Memprediksi Intensi Pembelian dari Pola Perilaku 885K+ Event eCommerce")
    st.markdown("")

    # ── SCQR STRUCTURE ──
    scqr_box("S",
        "Conversion rate e-commerce hanya 2–5%.",
        "Dari 490,398 sesi browsing di electronics store ini, hanya 24,344 (4.96%) "
        "yang berakhir dengan pembelian. 95% traffic tidak menghasilkan revenue.")

    scqr_box("C",
        "Tim marketing tidak bisa membedakan siapa yang akan beli.",
        "Tanpa kemampuan membedakan high-intent vs low-intent session, retargeting "
        "dilakukan secara broadcast ke semua user — over-spend pada yang tidak akan beli, "
        "under-invest pada yang sebenarnya siap membeli.")

    scqr_box("Q",
        "Bisakah kita memprediksi probabilitas pembelian dari pola perilaku browsing?",
        "Bisakah model machine learning mengidentifikasi sesi mana yang paling mungkin "
        "berakhir dengan purchase — sebelum keputusan itu terjadi?")

    scqr_box("R",
        "Ya. CatBoost mencapai AUC-ROC 0.9953 — hanya perlu target 6.3% sesi untuk tangkap 91.5% buyers.",
        "Dengan threshold 0.85: precision 80%, recall 91.5%. Dari setiap 100 sesi yang di-flag, "
        "80 benar-benar membeli. Model ini memungkinkan retargeting yang 15× lebih efisien "
        "dibanding broadcast.")

    st.markdown("")

    # ── LEVEL 1: KEY METRICS ──
    st.markdown('<div class="pyramid-l1">', unsafe_allow_html=True)
    st.markdown("#### 📐 Level 1 — Bottom Line Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("0.9953", "AUC-ROC", "95% CI: 0.9948–0.9957")
    with c2: metric_card("0.9382", "PR-AUC", "Imbalanced-aware")
    with c3: metric_card("0.854", "F1-Score", "@ threshold 0.85")
    with c4: metric_card("6.3%", "Target Rate", "untuk 91.5% recall")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── LEVEL 2: THREE KEY FINDINGS ──
    st.markdown("#### 📐 Level 2 — Tiga Temuan Kunci")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="pyramid-l2">', unsafe_allow_html=True)
        st.markdown("**A. User History > Session Signals**")
        st.markdown("""
        `events_per_session` (31.1%) dan `total_sessions` (12.7%)
        mendominasi prediksi. **Siapa user-nya** lebih prediktif dari
        **apa yang dilakukan di sesi ini.**
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="pyramid-l2">', unsafe_allow_html=True)
        st.markdown("**B. Threshold 0.85 ≠ Default 0.5**")
        st.markdown("""
        Menaikkan threshold dari 0.5 ke 0.85 meningkatkan precision dari
        68.3% → 80.0% dengan recall hanya turun dari 96.8% → 91.5%.
        Mengurangi wasted budget signifikan.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="pyramid-l2">', unsafe_allow_html=True)
        st.markdown("**C. 5 Model, CatBoost Menang**")
        st.markdown("""
        Semua 5 tree-based models >0.99 AUC. CatBoost unggul tipis
        (0.9948) berkat ordered boosting & auto class weights.
        Overfit gap hanya 0.0014.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════
# PAGE 2: DATA & FUNNEL (Level 2A — Data Evidence)
# ══════════════════════════════════════════════════════════════
elif page == "📊 Data & Funnel":

    st.markdown("# 📊 Data & Conversion Funnel")
    st.markdown("##### Level 2A — Bukti dari data yang mendasari model")

    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **FINDING:** Dari 793,589 event view, hanya 54,029 (6.81%) menjadi cart,
    dan 37,346 (4.71%) menjadi purchase. Tapi begitu user memasukkan produk ke cart,
    **69.12% akan membeli** — cart event adalah momen paling kritis.""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # Funnel chart
    col1, col2 = st.columns([3, 2])
    with col1:
        fig = go.Figure(go.Funnel(
            y=["View (793,589)", "Add to Cart (54,029)", "Purchase (37,346)"],
            x=[793589, 54029, 37346],
            textinfo="value+percent initial",
            marker_color=["#2563EB", "#D97706", "#059669"],
            connector={"line": {"color": "#475569"}},
        ))
        fig.update_layout(
            title="Conversion Funnel",
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F1F5F9"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("")
        st.markdown("")
        c1, c2 = st.columns(2)
        with c1: metric_card("6.81%", "View → Cart")
        with c2: metric_card("69.12%", "Cart → Purchase")
        st.markdown("")
        c3, c4 = st.columns(2)
        with c3: metric_card("4.96%", "Purchase Rate", "Session-level")
        with c4: metric_card("19.1:1", "Imbalance", "Non-purchase : Purchase")

    st.markdown("---")

    # Dataset overview
    st.markdown("#### Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("885,129", "Total Events", "Sep 2020 — Feb 2021")
    with c2: metric_card("490,398", "Unique Sessions", "Unit of prediction")
    with c3: metric_card("407,283", "Unique Users", "Electronics store")
    with c4: metric_card("53,453", "Products", "Multiple categories")

    st.markdown("---")

    st.markdown("#### Data Quality")
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **Missing values:** `category_code` 26.69%, `brand` 23.99% — keduanya hanya untuk EDA, bukan fitur model.
    **Outlier harga:** 66,308 (7.5%) berdasarkan IQR — TIDAK dihapus karena valid secara bisnis
    (produk premium) dan tree-based models robust terhadap outlier.
    **Duplikasi:** 655 rows (0.07%) — minimal.""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Temporal & Price insights
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Temporal Patterns")
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        **Peak hour:** 10:00 pagi (4.78% purchase rate).
        Weekday (4.25%) ≈ Weekend (4.15%) — electronics purchase merata sepanjang minggu.
        Feature temporal menjadi moderate predictor (bukan top, tapi memberikan sinyal tambahan).""")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### Price Behavior")
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        **Cart median $82.97 > Purchase median $64.48.**
        User memasukkan produk lebih mahal ke cart tapi membeli yang sedikit lebih murah.
        Browsing bersifat eksplorasi luas (std $307), keputusan beli lebih terfokus (std $170).""")
        st.markdown('</div>', unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════
# PAGE 3: MODEL ARENA (Level 2B — Model Comparison)
# ══════════════════════════════════════════════════════════════
elif page == "🏆 Model Arena":

    st.markdown("# 🏆 Model Arena — 5 Tree-Based Models")
    st.markdown("##### Level 2B — Perbandingan performa dan justifikasi pemilihan model")

    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **FINDING:** Semua 5 model mencapai AUC-ROC >0.99. **CatBoost menang** (0.9948)
    berkat ordered boosting dan auto class weights. Setelah tuning, CatBoost mencapai
    **0.9953** dengan overfit gap hanya **0.0014** — model sangat stabil.
    Imbalance ditangani melalui mekanisme built-in masing-masing model, bukan SMOTE.""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # Model comparison table
    df_display = df_models.copy()
    df_display.index.name = "Model"

    # Highlight best
    st.dataframe(
        df_display.style
            .format({"AUC-ROC": "{:.4f}", "PR-AUC": "{:.4f}", "F1-Score": "{:.4f}",
                      "Precision": "{:.4f}", "Recall": "{:.4f}"})
            .background_gradient(subset=["AUC-ROC"], cmap="Greens")
            .background_gradient(subset=["PR-AUC"], cmap="Blues"),
        use_container_width=True,
    )

    st.markdown("")

    # Visual comparison
    col1, col2 = st.columns(2)

    with col1:
        models_list = df_models.index.tolist()
        colors = ["#10B981" if "Tuned" in m else "#2563EB" for m in models_list]

        fig = go.Figure(go.Bar(
            x=df_models["AUC-ROC"].values,
            y=models_list,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in df_models["AUC-ROC"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="AUC-ROC Comparison",
            xaxis_range=[0.985, 1.0],
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F1F5F9"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=df_models["PR-AUC"].values,
            y=models_list,
            orientation="h",
            marker_color=["#10B981" if "Tuned" in m else "#7C3AED" for m in models_list],
            text=[f"{v:.4f}" for v in df_models["PR-AUC"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="PR-AUC Comparison (Imbalanced-Aware)",
            xaxis_range=[0.86, 0.95],
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F1F5F9"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Imbalance handling explainer
    st.markdown("#### Strategi Penanganan Class Imbalance (19.1:1)")
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **Mengapa BUKAN SMOTE?** Synthetic oversampling pada data behavioral menciptakan
    "sesi belanja fiktif" yang tidak pernah terjadi di kenyataan. Semua model menggunakan
    mekanisme built-in yang lebih stabil:

    | Model | Mekanisme | Cara Kerja |
    |---|---|---|
    | Random Forest | `class_weight='balanced'` | Penalti lebih besar untuk miss kelas minoritas |
    | Gradient Boosting | Sequential correction | Iterasi berikutnya fokus pada error sebelumnya |
    | XGBoost | `scale_pos_weight=19.67` | Setiap sample purchase dihitung 19.67× |
    | LightGBM | `scale_pos_weight=19.67` | Sama dengan XGBoost, lebih cepat |
    | CatBoost | `auto_class_weights='Balanced'` | Auto-calculate optimal class weights |
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Tuning results
    st.markdown("#### CatBoost Tuning Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("0.9953", "Test AUC-ROC", "After tuning")
    with c2: metric_card("+0.0005", "Improvement", "vs baseline CatBoost")
    with c3: metric_card("0.0014", "Overfit Gap", "Train 0.9982 vs CV 0.9968")
    with c4: metric_card("30", "Iterations", "RandomizedSearchCV, 5-fold")

    st.markdown("")
    with st.expander("Best Hyperparameters"):
        st.json({
            "iterations": 300,
            "learning_rate": 0.1,
            "depth": 7,
            "l2_leaf_reg": 3,
            "bagging_temperature": 2,
        })

    footer()


# ══════════════════════════════════════════════════════════════
# PAGE 4: WHY IT WORKS (Level 2C — Feature Importance + SHAP)
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Why It Works":

    st.markdown("# 🔍 Why It Works — Model Interpretation")
    st.markdown("##### Level 2C — Feature importance, SHAP, threshold, dan error analysis")

    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **FINDING:** Berbeda dari hipotesis awal yang memprediksi session intent signals (cart, view_to_cart_ratio)
    sebagai top predictor, justru **user history** yang mendominasi: `events_per_session` (31.1%),
    `total_sessions` (12.7%), `is_repeat_buyer` (6.1%). **Siapa user-nya lebih prediktif
    dari apa yang dilakukan di sesi ini.**""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # Feature importance chart
    st.markdown("#### Feature Importance — Top 10")
    fig = go.Figure(go.Bar(
        x=df_features["Importance"].values[::-1],
        y=df_features["Feature"].values[::-1],
        orientation="h",
        marker_color=["#2563EB" if i > 2 else "#10B981"
                       for i in range(len(df_features)-1, -1, -1)],
        text=[f"{v:.1f}%" for v in df_features["Importance"].values[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F1F5F9"),
        xaxis_title="Importance (%)",
        margin=dict(l=10, r=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Business interpretation table
    st.markdown("#### Business Interpretation & Actionable Recommendations")

    for _, row in df_features.iterrows():
        with st.expander(f"**#{int(row['Rank'])} — `{row['Feature']}`** (importance: {row['Importance']:.1f}%)"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**💡 Business Meaning:**")
                st.markdown(row["Business Meaning"])
            with col2:
                st.markdown("**🎯 Actionable Recommendation:**")
                st.markdown(row["Actionable Recommendation"])

    st.markdown("---")

    # Threshold analysis
    st.markdown("#### Threshold Analysis")
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **Threshold TIDAK mengubah model** — hanya mengubah aturan keputusan bisnis.
    Model menghasilkan probability score (0–1) yang sudah fixed.
    Threshold menentukan: "di atas berapa kita TINDAK LANJUTI?"
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Threshold slider
    thresholds_data = {
        0.30: {"P": 0.561, "R": 0.978, "F1": 0.713, "pct": 9.6},
        0.50: {"P": 0.683, "R": 0.968, "F1": 0.801, "pct": 7.8},
        0.65: {"P": 0.728, "R": 0.957, "F1": 0.827, "pct": 7.2},
        0.75: {"P": 0.761, "R": 0.946, "F1": 0.844, "pct": 6.8},
        0.85: {"P": 0.800, "R": 0.915, "F1": 0.854, "pct": 6.3},
    }

    thresh_val = st.select_slider(
        "Pilih threshold:",
        options=list(thresholds_data.keys()),
        value=0.85,
        format_func=lambda x: f"{x:.2f}",
    )

    td = thresholds_data[thresh_val]
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card(f"{td['P']*100:.1f}%", "Precision", "Dari flag, berapa yang benar beli")
    with c2: metric_card(f"{td['R']*100:.1f}%", "Recall", "Dari buyers, berapa yang tertangkap")
    with c3: metric_card(f"{td['F1']:.3f}", "F1-Score", "Harmonic mean P & R")
    with c4: metric_card(f"{td['pct']}%", "% Sesi Di-flag", f"dari 98,127 test sessions")

    if thresh_val == 0.85:
        st.success("✅ **Threshold optimal** — F1 tertinggi (0.854) dengan precision 80%.")
    elif thresh_val < 0.85:
        st.info(f"ℹ️ Threshold lebih rendah = tangkap lebih banyak buyers, tapi lebih banyak false positive.")
    else:
        st.warning(f"⚠️ Threshold lebih tinggi = precision naik, tapi recall turun signifikan.")

    st.markdown("---")

    # Error Analysis
    st.markdown("#### Error Analysis")
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **False Negatives (456 missed buyers):** Median `events_per_session` = 2.33,
    `max_price_viewed` = $257. Ini adalah **user aktif yang browsing produk mahal
    tapi model kurang yakin** — kemungkinan first-time buyers tanpa purchase history.

    **False Positives (1,231 wrong flags):** Median `events_per_session` = 2.33,
    `max_price_viewed` = $210. Profil mirip false negatives — model kesulitan
    di "grey zone" antara active browsers dan actual buyers.

    **Improvement:** Tambah fitur category-level intent dan product price tier
    untuk membedakan browsing eksplorasi vs purposeful.""")
    st.markdown('</div>', unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════
# PAGE 5: PREDICT NOW (Level 3 — Interactive Simulator)
# ══════════════════════════════════════════════════════════════
elif page == "🎯 Predict Now":

    st.markdown("# 🎯 Predict Now — Purchase Probability Simulator")
    st.markdown("##### Level 3 — Coba model secara interaktif")

    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    Masukkan parameter sesi browsing di bawah ini. Model CatBoost (Tuned) akan
    menghitung **probability score** — seberapa mungkin sesi ini berakhir dengan purchase.""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    try:
        model = load_model()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.warning(f"Model file tidak ditemukan atau gagal dimuat. Simulator menggunakan mode demo.")

    # Input form
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🖱️ Session Behavior**")
        n_events = st.slider("Total events di sesi ini", 1, 50, 5)
        n_view = st.slider("Jumlah view events", 1, 50, 4, help="Biasanya ≈ n_events")
        n_cart = st.slider("Jumlah cart events", 0, 10, 1)
        n_unique_products = st.slider("Produk unik dilihat", 1, 30, 3)
        n_unique_categories = st.slider("Kategori unik", 1, 10, 2)
        n_unique_brands = st.slider("Brand unik", 1, 15, 2)

    with col2:
        st.markdown("**💰 Price Signals**")
        avg_price = st.number_input("Avg price viewed ($)", 1.0, 5000.0, 65.0)
        max_price = st.number_input("Max price viewed ($)", 1.0, 65000.0, 150.0)
        min_price = st.number_input("Min price viewed ($)", 0.1, 5000.0, 30.0)
        avg_price_carted = st.number_input("Avg price carted ($)", 0.0, 5000.0, 80.0 if n_cart > 0 else 0.0)
        total_cart_value = st.number_input("Total cart value ($)", 0.0, 50000.0, avg_price_carted * n_cart)

    with col3:
        st.markdown("**👤 User History**")
        total_sessions = st.slider("Total sesi historis user", 1, 50, 2)
        is_repeat_buyer = st.selectbox("Pernah beli sebelumnya?", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        events_per_session = st.number_input("Avg events per session", 0.1, 100.0, float(n_events) / max(total_sessions, 1))
        recency_days = st.slider("Hari sejak terakhir aktif", 0, 180, 10)
        tenure_days = st.slider("Hari sejak pertama terdaftar", 0, 200, 30)
        hour = st.slider("Jam sesi dimulai", 0, 23, 10)
        dow = st.slider("Hari (0=Mon, 6=Sun)", 0, 6, 2)

    st.markdown("---")

    # Compute derived features
    view_to_cart_ratio = n_cart / (n_view + 1)
    has_cart = int(n_cart > 0)
    browse_to_action = n_cart / (n_events + 1)
    product_per_event = n_unique_products / (n_events + 1)
    session_duration = min(n_events * 0.8, 480)
    is_weekend = int(dow >= 5)
    std_price = abs(max_price - min_price) / 3 if max_price > min_price else 0
    price_range = max_price - min_price
    n_cart_items = n_cart
    cart_view_ratio = avg_price_carted / (avg_price + 0.01) if avg_price > 0 else 0

    features = np.array([[
        n_events, n_view, n_cart, n_unique_products, n_unique_categories,
        n_unique_brands, view_to_cart_ratio, has_cart, browse_to_action,
        product_per_event, session_duration, hour, dow, is_weekend,
        avg_price, max_price, min_price, std_price, price_range,
        avg_price_carted, total_cart_value, n_cart_items, cart_view_ratio,
        recency_days, tenure_days, total_sessions, is_repeat_buyer,
        events_per_session,
    ]])

    # Predict
    if st.button("🔮 Prediksi Purchase Probability", type="primary", use_container_width=True):
        if model_loaded:
            try:
                proba = model.predict_proba(features)[0][1]
            except Exception:
                proba = model.predict_proba(features.reshape(1, -1))[0][1]
        else:
            # Demo mode: simple heuristic
            score = (has_cart * 0.3 + is_repeat_buyer * 0.2 + min(events_per_session / 5, 0.2)
                     + min(total_sessions / 10, 0.15) + min(n_cart / 3, 0.15))
            proba = min(max(score, 0.02), 0.98)

        threshold = cfg.get("optimal_threshold", 0.85)
        prediction = "🟢 LIKELY TO PURCHASE" if proba >= threshold else "🔴 UNLIKELY TO PURCHASE"

        st.markdown("---")
        st.markdown("### Hasil Prediksi")

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 48, "color": "#F1F5F9"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#64748B"},
                    "bar": {"color": "#10B981" if proba >= threshold else "#EF4444"},
                    "bgcolor": "#1E293B",
                    "steps": [
                        {"range": [0, threshold * 100], "color": "#1E293B"},
                        {"range": [threshold * 100, 100], "color": "rgba(16,185,129,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#F59E0B", "width": 3},
                        "thickness": 0.8,
                        "value": threshold * 100,
                    },
                },
            ))
            fig.update_layout(
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F1F5F9"),
                margin=dict(t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            metric_card(f"{proba:.1%}", "Probability", "Purchase probability")
        with c3:
            if proba >= threshold:
                st.markdown(f"""
                <div class="metric-card" style="border-color:#10B981;">
                    <div class="metric-value" style="color:#10B981;">✅</div>
                    <div class="metric-label">LIKELY BUYER</div>
                    <div class="metric-sub">Above threshold {threshold}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card" style="border-color:#EF4444;">
                    <div class="metric-value" style="color:#EF4444;">❌</div>
                    <div class="metric-label">UNLIKELY</div>
                    <div class="metric-sub">Below threshold {threshold}</div>
                </div>""", unsafe_allow_html=True)

        # Recommendation
        st.markdown("")
        if proba >= threshold:
            st.markdown('<div class="insight-box" style="border-color:#10B981;">', unsafe_allow_html=True)
            st.markdown(f"""
            **Rekomendasi:** Sesi ini memiliki probabilitas purchase **{proba:.1%}** — di atas threshold {threshold}.
            **Tindakan:** Kirim targeted offer, tampilkan limited-time discount,
            atau trigger personalized push notification untuk mendorong konversi segera.""")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-box" style="border-color:#EF4444;">', unsafe_allow_html=True)
            st.markdown(f"""
            **Rekomendasi:** Sesi ini memiliki probabilitas purchase **{proba:.1%}** — di bawah threshold {threshold}.
            **Tindakan:** Jangan alokasikan budget retargeting mahal.
            Pertimbangkan soft nurturing (email newsletter, wishlist reminder) untuk engagement jangka panjang.""")
            st.markdown('</div>', unsafe_allow_html=True)

    footer()
