from typing import Iterable

from numpy import ndarray, var
from scipy.stats import entropy

def variance_gain(common: Iterable[ndarray[float]],
                  left: Iterable[ndarray[float]],
                  right: Iterable[ndarray[float]]):
  left_p = len(left) / len(common)
  right_p = len(right) / len(common)
  return var(common) - left_p * var(left) - right_p * var(right)

def information_gain(common: Iterable[ndarray[float]],
                     left: Iterable[ndarray[float]],
                     right: Iterable[ndarray[float]]):
  left_p = len(left) / len(common)
  right_p = len(right) / len(common)
  return entropy(common) - left_p * entropy(left) - right_p * entropy(right)
