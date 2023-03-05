from typing import Iterable, overload
from dataclasses import dataclass

from numpy import ndarray, array
import numpy as np

@dataclass
class Node(object):
  feature: int = None
  threshold: float = None
  left: 'Node' = None
  right: 'Node' = None
  variance: float = None
  value: float = None

  def is_leaf(self): return self.value is not None

  def predict(self, rows: Iterable[ndarray[float]]) -> ndarray[float]:
    return array([self._predict(row) for row in rows])

  def present(self, indentation=0):
    if self.is_leaf():
      yield f"{self.value}\n"
      return
    yield f"X{self.feature} <= {self.threshold} ? {self.variance:.4f}\n"
    yield f"{'': >{indentation}} │ L - "
    yield from self.left.present(indentation + 2)
    yield f"{'': >{indentation}} └ R - "
    yield from self.right.present(indentation + 2)

  def _predict(self, row: ndarray[float]) -> float:
    if self.is_leaf(): return self.value

    if row[self.feature] <= self.threshold:
      return self.left._predict(row)
    return self.right._predict(row)
