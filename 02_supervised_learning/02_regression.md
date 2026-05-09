# Regression

Regression predicts a continuous value (price, temperature, housing value).

## Linear Regression

Fits a straight line (or hyperplane) through the data by minimizing the sum of squared errors (OLS). No hyperparameters.

**Metrics:**

| Metric | Formula | Notes |
|---|---|---|
| MSE | mean of (y - ŷ)² | penalizes large errors heavily |
| MAE | mean of |y - ŷ| | more robust to outliers |
| R² | 1 - SS_res/SS_tot | 1.0 = perfect, 0 = baseline mean model |

## Regularization

Adds a penalty to the loss to discourage large weights → reduces overfitting.

**Lasso (L1):** penalty = α × Σ|wᵢ|
- Drives some weights exactly to zero → built-in feature selection
- Hyperparameter: `alpha` (strength)

**Ridge (L2):** penalty = α × Σwᵢ²
- Shrinks all weights smoothly, none go to zero
- Better when all features are relevant

**ElasticNet:** combines L1 and L2
- `alpha` controls overall strength
- `l1_ratio` controls the mix (0 = Ridge, 1 = Lasso)

## Pipelines

Always wrap preprocessing + model in a Pipeline to avoid data leakage:

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])
```

The pipeline fits the scaler only on training data and applies the same transform to the test set.

## Hyperparameter Tuning

**GridSearchCV:** exhaustive search over a defined parameter grid with k-fold CV.

**RandomizedSearchCV:** samples a fixed number of combinations at random — faster for large grids.

```python
GridSearchCV(pipeline, {'ridge__alpha': [0.1, 1, 10]}, cv=5)
RandomizedSearchCV(pipeline, {'ridge__alpha': np.arange(0, 10, 0.01)}, cv=5, n_iter=100)
```

Always select hyperparameters using validation/CV performance, never the test set.
