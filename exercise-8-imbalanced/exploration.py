from matplotlib import pyplot as plt
import dataset as ds
import seaborn as sns

def main():
  df = ds.load()
  print(df.info())
  print(df.describe())

  plt.figure(figsize=(120, 16))
  sns.boxplot(data=df)
  plt.xticks(rotation=90)
  plt.show()

if __name__ == '__main__':
  main()
