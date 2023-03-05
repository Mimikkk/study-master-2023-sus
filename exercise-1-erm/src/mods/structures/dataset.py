from numpy import ndarray, array

DataSet = tuple[ndarray[float], ndarray[float]]
def split(dataset: DataSet, split_at=0.2) -> tuple[DataSet, DataSet]:
  (x, y) = dataset
  n = len(x)

  n_test = int(n * split_at)
  n_train = n - n_test

  x_train, x_test = x[:n_train], x[n_train:]
  y_train, y_test = y[:n_train], y[n_train:]
  return (x_train, y_train), (x_test, y_test)
