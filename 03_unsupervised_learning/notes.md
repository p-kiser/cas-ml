# Unsupervised Learning

Unsupervised learning is used with unlabeled data, meaning we do not know the outcome. Therefore it is not used to make predictions. Typical use cases:

- clustering
- anomaly detection
- dimensionality reduction
- visualization

Examples: customer segmentation in marketing, fraud detection in insurance, facial recognition...

## Clustering

Clustering is a fundamental task in unsupervised learning. It helps understanding the underlying structure of data.

The goal of clustering is to group similar data points based on their inherent characteristics (similar features). Objects in the same cluster should be more similar to each other than to those of other clusters.

## Anomaly Detection

The goal is to identify unusual patterns in a dataset and identify data points that are dissimilar from the majority.

By clustering the data, we can establish a notion of what is considered normal or expected in each cluster.

## Distance

- **Euclidian Distance**: literal, exact distance, but computationally expensive
- **Simple Matching Distance**: 0 if the same, 1 if not the same
- **Dynamic Time Warping**: for time series
- **Manhattan**
- **Cosine**

## K-means Clustering

Clustering algorithm:

1. Assign each data point to the closest cluster center
2. Set each cluster center as the mean of the data points assignedto it
3. Stop when there are no more new assignments

The number of clusters (k) is fixed by the analyst as a hyperparameter.

## Optimization

A good clustering is one for which the variation within a cluster $C_k$ is as small as possible. The goal of optimization is to minimize this.

This means scaling is very important for clusteering any *any* numeric value.


