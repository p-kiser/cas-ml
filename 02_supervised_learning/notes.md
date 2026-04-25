---
title: Supervised Learning
---

# Recap

**Gradient Descent**

Iteratively adjust model parameters by stepping in the direction that reduces error most. The learning rate controls step size.

**Linear Regression**

Fit a straight line through data to predict a continuous value. Finds the line that minimizes the sum of squared errors.

**Imputation**

Fill in missing values with substitutes — e.g. mean, median, or a model-predicted value — rather than dropping rows.

**Data Scaling**

Bring features to a comparable range (e.g. 0–1 or zero mean) so no feature dominates due to its unit size. Required for distance-based and gradient methods.

**Loss Functions**

A measure of how wrong the model's predictions are. Training minimizes this. E.g. MSE for regression, cross-entropy for classification.

**Feature Selection**

Choose only the input variables that actually help predict the target. Reduces noise, overfitting, and training cost.

**Bias-Variance**

High bias (underfitting): model too simple, wrong on both train and test
High variance (overfitting): model too complex, good on train, bad on test
Diagnose by comparing train vs. test error — large gap = overfitting, both high = underfitting

**Train / Test / Validation & Cross-Validation**

- Train: fit the model
- Validation: tune hyperparameters
- Test: final unbiased evaluation
- Cross-validation: rotate the validation split across k folds to get a more reliable estimate without wasting data

**Regularisation**

Penalize model complexity during training to reduce overfitting. L1 (Lasso) drives weights to zero (feature selection). L2 (Ridge) shrinks weights smoothly.

**Model Selection**

Choose between candidate models (or hyperparameters) based on validation performance, not test performance. Test is touched only once at the end.**

# Classification vs Regression

- Regression: predict continuous values (price, temperature)
- Classification: predict discrete classes (yes / no, red / blue / green)

Any regression can be converted to a classification problem by discretizing the target variable. This simplifies the modeling by reducing the complexity of the data.

For binary classification> use sigmoid (0-1) on linear output and use a threshold (e.g. 0.5)

More Infos:

https://machinelearningmind.wordpress.com/2019/11/25/logistic-regression-explained/


# Percicion and Recall

Measuring true positives, false positives, etc. by combining them into some metrics:

    Precision = true positives / all positive predictions

Measures how often we where correct (specificity).

    Recall = true positives / actual positives

Measures how many positives where detected (sensitivity).

    Accuracy = all correct predictions / all predictions

Accuracy can be misleading if the distrubution is imbalanced. If positives are rare (cancer, fraud detection etc), a model always predicting false has high accuracy, but a precision and recall of 0.

There are some others, like F1 Score (combination of precision and recall). This is often more useful than accuracy.

    F1 Score = (2 * TP) / (2 * TP + FP + FN)



