import pandas as pd

def load():
  return pd.read_csv('resources/datasets/145317-clustersel.txt', sep='\t', header=None)
