from matplotlib import pyplot as plt
import numpy as np

import dataset as ds
import seaborn as sns

def main():
  df = ds.load()
  print(df.head())
  print(df.info())
  print(df.describe())

  data = df.values


  X = data[:, :-1]
  y = data[:, -1].astype(int)

  vowels = "a-e-i-o-u-y-sz-z-ź-ż-g-r".split('-')
  labels = [
    f"{vowels[mod]}{div}"
    for value in y
    for (div, mod) in [divmod(value, 100)]
  ]

if __name__ == '__main__':
  main()
