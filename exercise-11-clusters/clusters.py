import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import dataset as ds
import seaborn as sns

def main():
  dataset: DataFrame = ds.load()
  X, y = dataset.values[:, :-1], dataset.values[:, -1].astype(int)
  # take log
  X = np.log(X)


  pca_2d = PCA(n_components=2)
  values_2d = pca_2d.fit_transform(X)
  explained_variance_2d = sum(pca_2d.explained_variance_ratio_)

  plt.figure(figsize=(10, 7))

  vowels = "a-e-i-o-u-y-sz-z-ź-ż-g-r".split('-')
  labels = [
    f"{vowels[mod]}{div}"
    for value in y
    for (div, mod) in [divmod(value, 100)]
  ]

  for i, pair in enumerate(values_2d):
    plt.text(pair[0], pair[1], labels[i])

  k = 12
  plt.title(f'2D PCA - KMeans-{k} (Wyjaśniona wariancja: {explained_variance_2d:.2%})')

  k_means = KMeans(init='k-means++', n_clusters=k, n_init=10)
  k_means.fit(X)
  clustercenters_trans = pca_2d.transform(k_means.cluster_centers_)

  plt.scatter(clustercenters_trans[:, 0], clustercenters_trans[:, 1], marker='x', color='red')

  unique_labels = set(k_means.labels_)
  color_map = matplotlib.colormaps['tab20']

  for label in unique_labels:
    cluster = pca_2d.transform(X[k_means.labels_ == label])
    plt.scatter(cluster[:, 0], cluster[:, 1], c=[color_map(label)], alpha=0.8)

  plt.show()


if __name__ == '__main__':
  main()
