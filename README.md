# 🛒 Purchase Probability Prediction — Streamlit Dashboard

## Framework
**SCQR Full Structure + Pyramid Principle per Page**

| Page | SCQR Role | Pyramid Level | Konten |
|---|---|---|---|
| 🏠 Executive Summary | S→C→Q→R | Level 1 (Bottom Line) | SCQR narrative + key metrics |
| 📊 Data & Funnel | Supporting | Level 2A (Evidence) | Conversion funnel, data quality, patterns |
| 🏆 Model Arena | Supporting | Level 2B (Model) | 5 model comparison, tuning results |
| 🔍 Why It Works | Supporting | Level 2C (Interpretation) | Feature importance, SHAP, threshold, errors |
| 🎯 Predict Now | Interactive | Level 3 (Data/Demo) | Real-time prediction simulator |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

```
streamlit_app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Theme configuration
├── model/
│   ├── catboost_tuned.pkl    # Trained CatBoost pipeline
│   └── model_config.json    # Model configuration & metrics
└── data/
    ├── model_comparison_final.csv
    └── feature_business_interpretation.csv
```

## Model Summary

- **Best Model:** CatBoost (Tuned)
- **AUC-ROC:** 0.9953 (95% CI: 0.9948–0.9957)
- **Optimal Threshold:** 0.85 → Precision 80%, Recall 91.5%
- **Features:** 28 engineered features from 3 groups (session intent, price signals, user history)

## Author

Reinaldi Santoso · Mining Engineering (ITB) · diBimbing Data Science Bootcamp
