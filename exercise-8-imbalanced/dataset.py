import pandas as pd

def load():
  return pd.read_csv('resources/datasets/145317-imbalanced.txt', sep='\t')
