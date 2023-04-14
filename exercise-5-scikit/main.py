import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
  columns = np.genfromtxt('resources/datasets/regression.txt', dtype=str, max_rows=1)
  rows = np.genfromtxt('resources/datasets/regression.txt', dtype=np.float32, skip_header=1)
  frame = pd.DataFrame(data=rows, columns=columns)
  # Analysis
  print(f"Number of attributes: {frame.shape[1] - 1}")
  print(f"Number of rows: {frame.shape[0]}")

  print(f"columns:")
  print([*frame.columns])

  print("Attribute Types:")
  print(frame.dtypes)

  print("Attribute Ranges:")
  print(frame.describe())

  n_columns = frame.shape[1]
  figure, axes = plt.subplots(2, n_columns, figsize=(120, 8))
  for (first, second), column in zip(zip(*axes), frame.columns):
    sns.boxplot(ax=first, data=frame[[column]])
    first.set_xticklabels(labels=[column])

    sns.violinplot(ax=second, data=frame[[column]])
    second.set_xticklabels(labels=[column])

  plt.tight_layout()
  plt.savefig(f'resources/figures/attribute-box-violin-plots.png')
  plt.show()


if __name__ == '__main__':
  main()
