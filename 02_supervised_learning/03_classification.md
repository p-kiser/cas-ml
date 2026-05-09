# Classification

Classification predicts a discrete class label (survived/died, digit 0–9, spam/not spam).

## Models

**Logistic Regression:** applies a sigmoid to a linear output to produce probabilities in [0, 1]. Threshold (e.g. 0.5) converts to a class. Regularization strength is controlled by `C` (inverse of lambda — smaller C = stronger regularization).

**Decision Tree:** splits data recursively on the feature that best separates classes. Hyperparameters: `max_depth`, `min_samples_leaf`. Can overfit deeply.

**Random Forest:** ensemble of many decision trees trained on random subsets of data and features. Averages predictions → lower variance, more robust. Hyperparameters: `n_estimators`, `max_depth`.

## Metrics

| Metric | Formula | When to use |
|---|---|---|
| Accuracy | correct / total | balanced classes |
| Precision | TP / (TP + FP) | when false positives are costly |
| Recall | TP / (TP + FN) | when false negatives are costly |
| F1 Score | 2·TP / (2·TP + FP + FN) | imbalanced classes |

Accuracy is misleading with imbalanced data. A model that always predicts "no cancer" has high accuracy but zero recall.

## Imbalanced Classes

When one class is much rarer (fraud, cancer), models learn to always predict the majority class.

**Oversampling (SMOTE):** generates synthetic samples for the minority class by interpolating between existing minority examples.

**Undersampling:** removes samples from the majority class.

**RandomOverSampler:** duplicates random minority samples (simpler than SMOTE).

Always apply resampling only to the training set, never the test set.

## Feature Importance

Random Forest provides `.feature_importances_` — how much each feature reduces impurity across all trees. Useful for identifying which inputs actually matter.

```python
importances = model.feature_importances_
sorted_idx = importances.argsort()
plt.barh(feature_names[sorted_idx], importances[sorted_idx])
```
