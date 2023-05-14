from pandas import DataFrame
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import dataset as ds

def main():
  dataset: DataFrame = ds.load()
  X, y = dataset.values[:, :-1], dataset.values[:, -1]

  pca_2d = PCA(n_components=2)
  values_2d = pca_2d.fit_transform(X)
  explained_variance_2d = sum(pca_2d.explained_variance_ratio_)

  plt.figure(figsize=(10, 7))
  scatter = plt.scatter(values_2d[:, 0], values_2d[:, 1], c=y, alpha=0.5)
  plt.legend(*scatter.legend_elements())
  plt.title(f'2D PCA (Wyjaśniona wariancja: {explained_variance_2d:.2%})')
  plt.show()

  pca_3d = PCA(n_components=3)
  values_3d = pca_3d.fit_transform(X)
  explained_variance_3d = sum(pca_3d.explained_variance_ratio_)

  df_3d = pd.DataFrame({'x': values_3d[:, 0], 'y': values_3d[:, 1], 'z': values_3d[:, 2], 'label': y})
  fig = plt.figure(figsize=(10, 7))
  ax = fig.add_subplot(111, projection='3d')

  ax.scatter(df_3d['x'], df_3d['y'], df_3d['z'], c=df_3d['label'], alpha=0.5)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.title(f'3D PCA (Wyjaśniona wariancja: {explained_variance_3d:.2%})')
  plt.show()


if __name__ == '__main__':
  main()
