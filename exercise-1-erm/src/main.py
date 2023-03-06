import numpy as np
from mods.utils import csv
from mods.structures import Node
import mods.structures.dataset as dataset

def mse(y, y_pred):
  return np.mean((np.array(y) - np.array(y_pred)) ** 2)
def rmse(y, y_pred):
  return np.sqrt(mse(y, y_pred))

if __name__ == '__main__':
  for identifier in map(str, range(1, 14)):
    ds = dataset.read(identifier)

    tree = Node.fit(ds, min_samples_per_split=int(len(ds) * 0.05), max_depth=5)
    print(*tree.present(), sep='')

    x_real = csv.read(f'{identifier}-test')
    csv.save(filename=identifier, solution=tree.predict(x_real))
