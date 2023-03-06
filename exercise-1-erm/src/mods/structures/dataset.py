from numpy import ndarray, concatenate
from numpy.random import shuffle
from ..utils import csv

DataSet = tuple[ndarray[float], ndarray[float]]
def split(dataset: DataSet, split_at=0.1) -> tuple[DataSet, DataSet]:
  (x, y) = dataset
  n = len(x)

  n_test = int(n * split_at)
  n_train = n - n_test

  x_train, x_test = x[:n_train], x[n_train:]
  y_train, y_test = y[:n_train], y[n_train:]
  return (x_train, y_train), (x_test, y_test)

def prepare(dataset: DataSet) -> DataSet:
  (x, y) = dataset
  ds = concatenate((x, y), axis=1)
  shuffle(ds)
  return ds[:, :-1], ds[:, -1]

def read(identifier: str) -> DataSet:
  return csv.read(f'{identifier}-X'), csv.read(f'{identifier}-Y')
