import matplotlib
import numpy as np
from pandas import DataFrame
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import dataset as ds
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN

def main():
  dataset: DataFrame = ds.load("cluster-contspeech")
  X, y = dataset.values[:, :-1], dataset.values[:, -1].astype(int)
  # X = np.log(X)

  pca_2d = PCA(n_components=2)
  values_2d = pca_2d.fit_transform(X)
  explained_variance_2d = sum(pca_2d.explained_variance_ratio_)

  # plt.figure(figsize=(10, 7))

  vowels = "a-e-i-o-u-y-sz-z-ź-ż-g-r".split('-')
  labels = [
    f"{vowels[mod]}{div}"
    for value in y
    for (div, mod) in [divmod(value, 100)]
  ]

  # k = 2
  # plt.title(f'2D KMeans=2 Log - (Wyjaśniona wariancja: {explained_variance_2d:.2%})')

  # k_means = KMeans(init='k-means++', n_clusters=k, n_init=10)
  # k_means.fit(X)

  # clustercenters_trans = pca_2d.transform(k_means.cluster_centers_)
  #
  # plt.scatter(clustercenters_trans[:, 0], clustercenters_trans[:, 1], marker='x', color='red')
  #
  # unique_labels = set(k_means.labels_)
  # color_map = matplotlib.colormaps['tab20']
  #
  # for label in unique_labels:
  #   print(label)
  #   cluster = pca_2d.transform(X[k_means.labels_ == label])
  #   plt.scatter(cluster[:, 0], cluster[:, 1], c=[color_map(label)], alpha=0.5)

  # plt.show()

  db = DBSCAN(eps=0.15)
  db.fit(values_2d)

  plt.figure(figsize=(10, 7))

  plt.title(f'2D DBSCAN - (Wyjaśniona wariancja: {explained_variance_2d:.2%})')

  unique_labels = set(db.labels_)
  print(db.labels_)
  color_map = matplotlib.colormaps['tab20']

  for label in unique_labels:
    print(label)
    cluster = pca_2d.transform(X[db.labels_ == label])
    plt.scatter(cluster[:, 0], cluster[:, 1], c=[color_map(label)], alpha=0.5)

  plt.show()


if __name__ == '__main__':
  main()
