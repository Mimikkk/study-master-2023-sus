import pandas as pd

def load(type="clustersel"):
  return pd.read_csv(f'resources/datasets/145317-{type}.txt', sep='\t', header=None)
