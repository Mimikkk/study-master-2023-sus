from sklearn.metrics import mean_squared_error
import numpy as np
from mods.utils import csv
from mods.structures import DecisionTreeRegressor
import mods.structures.dataset as dataset

if __name__ == '__main__':
  identifier = "1"

  ds = map(csv.read, (f'{identifier}-X', f'{identifier}-Y'))
  ds_train, (x_test, y_test) = dataset.split(ds)

  regressor = DecisionTreeRegressor()
  tree = regressor.fit(ds_train, min_samples_per_split=5, max_depth=5)
  print(*tree.present(), sep='')

  y_pred = tree.predict(x_test)
  err = np.sqrt(mean_squared_error(y_test, y_pred))

  print(f"RMSE is {err}")

  csv.save(filename=f'resources/results/{identifier}.csv', solution=y_pred)
  csv.save(
    filename=f'resources/results/{identifier}-com.csv',
    solution=[y - y_hat for (y, y_hat) in zip(y_test, y_pred)]
  )
