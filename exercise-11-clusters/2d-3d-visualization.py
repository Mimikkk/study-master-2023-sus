import numpy as np
from pandas import DataFrame
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import dataset as ds
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):
  counts = np.zeros(model.children_.shape[0])
  n_samples = len(model.labels_)
  for i, merge in enumerate(model.children_):
    current_count = 0
    for child_idx in merge:
      if child_idx < n_samples:
        current_count += 1
      else:
        current_count += counts[child_idx - n_samples]
    counts[i] = current_count

  linkage_matrix = np.column_stack(
    [model.children_, model.distances_, counts]
  ).astype(float)
  dendrogram(linkage_matrix, **kwargs)

def main():
  dataset: DataFrame = ds.load()
  X, y = dataset.values[:, :-1], dataset.values[:, -1].astype(int)

  vowels = "a-e-i-o-u-y-sz-z-ź-ż-g-r".split('-')
  labels = [
    f"{vowels[mod]}{div}"
    for value in y
    for (div, mod) in [divmod(value, 100)]
  ]

  pca_2d = PCA(n_components=2)
  values_2d = pca_2d.fit_transform(X)
  explained_variance_2d = sum(pca_2d.explained_variance_ratio_)

  model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
  model = model.fit(X)

  plot_dendrogram(model, labels=labels)
  plt.title(f'AgglomerativeClustering - Surowe (Wyjaśniona wariancja: {explained_variance_2d:.2%})')
  plt.show()

if __name__ == '__main__':
  main()
