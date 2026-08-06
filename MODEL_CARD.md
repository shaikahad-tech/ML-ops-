# Model Card — Customer Churn Classifier

## Model details
- **Model type:** RandomForestClassifier (scikit-learn Pipeline with preprocessing)
- **Task:** Binary classification — predict customer churn (1) vs. retention (0)
- **Version:** v1

## Intended use
- **Primary use:** Telecom customer-churn prediction for retention campaigns.
- **Out of scope:** Any production deployment without validation on real data. This model is trained on synthetic data and is intended as an MLOps reference project.

## Training data
Synthetically generated customer records (5,000 by default) with features:
`tenure_months`, `monthly_charges`, `total_charges`, `num_support_calls`, `num_products`, `contract_type`, `internet_service`, `payment_method`.

The churn label is derived from a logistic function of these features with added noise, calibrated to a ~26.5% positive rate.

## Evaluation metrics
| Metric    | Typical value (default config) |
|-----------|-------------------------------|
| Accuracy  | ~0.80                         |
| Precision | ~0.60                         |
| Recall    | ~0.70                         |
| F1        | ~0.65                         |
| ROC-AUC   | ~0.85                         |

(Metrics vary slightly by seed; run `python run.py train` to reproduce.)

## Ethical considerations
- The synthetic data does not represent real individuals.
- A production churn model can affect which customers receive retention offers. Validate fairness across demographic groups before deployment.
