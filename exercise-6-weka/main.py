import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
  csv = './results.csv'
  df = pd.read_csv(csv)
  print(df)

  x = range(len(df))
  df.plot.line(y='r2', use_index=True)
  df.plot.line(y='rmse', use_index=True)
  plt.show()
