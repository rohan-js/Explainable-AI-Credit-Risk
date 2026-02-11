# Explainable AI for Loan Default Risk Prediction
## Executive Summary

---

### Project Overview

This project implements a **production-grade Explainable AI system** for credit risk assessment. The system not only predicts loan defaults but provides transparent, auditable explanations for every decision—a requirement for regulatory compliance in financial services.

---

### Key Findings

#### 1. Predictive Performance
- **Best Model**: Random Forest (ROC-AUC: ~0.78)
- **Default Detection**: Identifies 70%+ of future defaults
- **Business Threshold**: Optimized for 5:1 cost ratio (false negatives 5x costlier than false positives)

#### 2. Top Risk Drivers (Global Analysis)
| Rank | Factor | Business Interpretation |
|------|--------|------------------------|
| 1 | Checking Account Status | No account = red flag |
| 2 | Loan Duration | Longer terms = higher risk |
| 3 | Credit History | Past behavior predicts future |
| 4 | Credit Amount | Larger loans = higher exposure |
| 5 | Savings Account | Financial buffer matters |

#### 3. Fairness Assessment
- **Gender**: Passes 80% rule (disparate impact compliant)
- **Age**: Young applicants rejected more often, but justified by actual default rates

---

### Business Impact

| Benefit | Description |
|---------|-------------|
| **Reduced Losses** | Earlier identification of high-risk applicants |
| **Regulatory Compliance** | Every decision explainable to regulators (ECOA, FCRA, GDPR Art. 22) |
| **Customer Trust** | Transparent reason codes for adverse actions |
| **Operational Efficiency** | Automated scoring with manual review escalation |

---

### Recommendations

1. **Deploy Random Forest model** with SHAP explanations for production scoring
2. **Implement continuous monitoring** for model drift and fairness degradation
3. **Establish quarterly retraining** cadence with updated loan outcomes
4. **Maintain audit documentation** including model cards and explanation logs

---

### Technical Approach

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Input    │────▶│   ML Models     │────▶│  Risk Score     │
│ (21 features)   │     │ (4 compared)    │     │  (0-100%)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                    ┌─────────────────┐
                    │  Explainability │
                    │  - SHAP (global)│
                    │  - LIME (local) │
                    └─────────────────┘
                                │
                                ▼
                    ┌─────────────────┐
                    │ Fairness Check  │
                    │ - Disparate     │
                    │   Impact        │
                    │ - Equalized Odds│
                    └─────────────────┘
```

---

### Dataset

- **Source**: South German Credit Dataset (UCI ML Repository)
- **Size**: 1,000 applicants, 21 features
- **Target**: Binary (Good Credit / Bad Credit)
- **License**: CC BY 4.0

---

*This report was generated as part of an Explainable AI portfolio project demonstrating production-grade risk modeling with regulatory-compliant explanations.*
