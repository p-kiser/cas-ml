# Dimensionality Reduction

High-dimensional data is hard to visualize and cluster. Dimensionality reduction compresses it into fewer dimensions while preserving structure.

## PCA (Principal Component Analysis)

**Linear** method. Finds the directions (principal components) of maximum variance in the data and projects onto them.

```python
pca = PCA(n_components=2)
pca.fit(df_scaled)
df_2d = pca.transform(df_scaled)
```

**Explained variance ratio:** how much of the total variance each component captures.

```python
pca.explained_variance_ratio_  # e.g. [0.19, 0.11] → first two PCs explain 30%
```

Use case: reduce 39 features → 2 dimensions to visualize clusters in a scatter plot. PCs are orthogonal (uncorrelated) linear combinations of original features.

**Key properties:**
- Fast and deterministic
- Good for linear structure
- Components are interpretable as weighted combinations of features

## t-SNE (t-distributed Stochastic Neighbor Embedding)

**Non-linear** method. Maps high-dimensional points to 2D/3D by preserving local neighborhoods — points close in high dimensions stay close in the projection.

**Key properties:**
- Great for visualization (reveals clusters and structure invisible to PCA)
- Non-deterministic: different runs may give different layouts
- Does not preserve global structure well (distances between clusters are not meaningful)
- Slow on large datasets
- Hyperparameter: `perplexity` (roughly how many neighbors to consider, typically 5–50)

```python
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
df_2d = tsne.fit_transform(df_scaled)
```

## PCA vs t-SNE

| | PCA | t-SNE |
|---|---|---|
| Type | Linear | Non-linear |
| Speed | Fast | Slow |
| Global structure | Preserved | Not preserved |
| Local structure | Partial | Strong |
| Deterministic | Yes | No |
| Use case | Compression, preprocessing | Visualization |

A common workflow: use PCA to reduce to ~50 dimensions first, then t-SNE on the reduced data for visualization.
