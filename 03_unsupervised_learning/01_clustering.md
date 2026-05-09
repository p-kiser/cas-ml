# Clustering

Clustering groups similar data points without labels. The goal: points within a cluster should be more similar to each other than to points in other clusters.

Use cases: customer segmentation, anomaly detection, data exploration.

## Distance Metrics

The choice of distance metric matters a lot:

| Metric | Description |
|---|---|
| Euclidean | Straight-line distance. Computationally expensive in high dimensions. |
| Manhattan | Sum of absolute differences per dimension. |
| Cosine | Angle between vectors. Good for text/embeddings. |
| Simple Matching | 0 if same, 1 if different. For binary features. |
| Dynamic Time Warping | For time series with different speeds. |

Scaling is always required before clustering — features with larger units will dominate distance calculations.

## K-Means

Algorithm:
1. Randomly place k cluster centers
2. Assign each point to the nearest center
3. Move each center to the mean of its assigned points
4. Repeat until assignments stop changing

**Hyperparameter:** k (number of clusters) — must be chosen in advance.

**Optimization goal:** minimize within-cluster variation (inertia):

```
minimize Σ_k Σ_{x ∈ Cₖ} ||x - μₖ||²
```

**Choosing k — Elbow Method:** plot inertia vs. k, look for the "elbow" where adding more clusters stops giving much benefit.

```python
km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, random_state=0)
km.fit(df_scaled)
labels = km.predict(df_scaled)
```

**Interpreting results:** group by cluster label and compute descriptive statistics (`groupby('cluster').describe()`). Boxplots help visualize differences between clusters per feature.

## Hierarchical Clustering

Builds a tree of clusters (dendrogram) without needing to specify k in advance.

**Agglomerative (bottom-up):** starts with each point as its own cluster, repeatedly merges the two closest clusters.

**Linkage methods** define "distance between clusters":
- `ward`: minimizes variance increase on merge (usually best)
- `average`: average distance between all pairs
- `complete`: maximum distance between any pair

```python
model = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
labels = model.fit_predict(df_scaled)
```

Read the dendrogram: cut it at a horizontal level to get clusters. A large vertical gap = a natural break point.

## Anomaly Detection via Clustering

Points far from any cluster center are anomalies. Establish "normal" via clusters, then flag points with high distance to their nearest center.
