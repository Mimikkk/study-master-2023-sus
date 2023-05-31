from pandas import DataFrame
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import dataset as ds

def main():
  dataset: DataFrame = ds.load()
  X, y = dataset.values[:, :-1], dataset.values[:, -1].astype(int)

  pca_2d = PCA(n_components=2)
  values_2d = pca_2d.fit_transform(X)
  explained_variance_2d = sum(pca_2d.explained_variance_ratio_)

  plt.figure(figsize=(10, 7))
  scatter = plt.scatter(values_2d[:, 0], values_2d[:, 1], c=y, alpha=0.5)
  plt.legend(*scatter.legend_elements())

  vowels = "a-e-i-o-u-y-sz-z-ź-ż-g-r".split('-')
  labels = [
    f"{vowels[mod]}{div}"
    for value in y
    for (div, mod) in [divmod(value, 100)]
  ]

  for i, pair in enumerate(values_2d):
    plt.text(pair[0], pair[1], labels[i])

  plt.title(f'2D PCA (Wyjaśniona wariancja: {explained_variance_2d:.2%})')
  plt.show()

if __name__ == '__main__':
  main()
