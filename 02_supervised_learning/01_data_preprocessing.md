# Data Preprocessing

## Train / Test / Validation Split

Always split before any preprocessing to avoid leakage:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

- **Train:** fit the model
- **Validation / CV:** tune hyperparameters
- **Test:** final evaluation, touched once

**Cross-validation:** splits training data into k folds, trains k times each with a different fold as validation. More reliable estimate than a single split.

## Imputation

Fill missing values using statistics from the training set only:

```python
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)           # learn median from train
X_train = imputer.transform(X_train)
X_test  = imputer.transform(X_test)  # apply same median to test
```

Never fit the imputer on test data — that would leak information.

## Feature Scaling

Required for distance-based methods (KNN, SVM) and gradient methods (logistic regression, neural networks). Not required for tree-based methods.

```python
scaler = StandardScaler()       # zero mean, unit variance
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)
```

`MinMaxScaler` scales to [0, 1] — useful for neural networks.

## Feature Engineering

Creating new features from existing ones:

- Binning continuous variables: `pd.cut(data['MedInc'], bins=[...])`
- Combining columns: `df['FamilySize'] = df['SibSp'] + df['Parch'] + 1`
- One-hot encoding categoricals: `pd.get_dummies(df, columns=['Sex', 'Embarked'])`

## Feature Selection

Remove features that don't help:

- Drop highly correlated features (check correlation matrix)
- Use Lasso — it zeros out irrelevant weights automatically
- Use Random Forest feature importances as a guide

## Correlation Matrix

```python
sns.heatmap(df.corr(), annot=True, cmap='Reds')
```

High correlation between two features = redundant information. Keep one or engineer a combined feature.
