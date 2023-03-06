from typing import Iterable
from dataclasses import dataclass
from ..utils import math
from numpy import ndarray, array, concatenate, unique, mean, shape

@dataclass
class Node(object):
  feature: int = None
  threshold: float = None
  left: 'Node' = None
  right: 'Node' = None
  variance: float = None
  value: float = None

  def is_leaf(self):
    return self.value is not None

  def predict(self, rows: Iterable[ndarray[float]]) -> ndarray[float]:
    return array([self.__predict(row) for row in rows])

  def present(self, indentation=0):
    if self.is_leaf():
      yield f"{self.value}\n"
      return
    yield f"X{self.feature} <= {self.threshold} ? {self.variance:.4f}\n"
    yield f"{'': >{indentation}} │ L - "
    yield from self.left.present(indentation + 2)
    yield f"{'': >{indentation}} └ R - "
    yield from self.right.present(indentation + 2)

  def __predict(self, row: ndarray[float]) -> float:
    if self.is_leaf(): return self.value

    if row[self.feature] <= self.threshold:
      return self.left.__predict(row)
    return self.right.__predict(row)

  @classmethod
  def __create_tree(cls, dataset, min_samples_per_split, max_depth, depth=0):
    values, labels = dataset[:, :-1], dataset[:, -1]
    samples, features = shape(values)
    print(samples, features, depth, min_samples_per_split, max_depth)
    if samples >= min_samples_per_split and depth <= max_depth:
      print(samples, features, depth)
      best = find_best_split(dataset)
      if best["variance"] > 0:
        left = cls.__create_tree(best["dataset_left"], min_samples_per_split, max_depth, depth + 1)
        right = cls.__create_tree(best["dataset_right"], min_samples_per_split, max_depth, depth + 1)
        return Node(
          feature=best["feature"],
          threshold=best["threshold"],
          left=left,
          right=right,
          variance=best["variance"]
        )

    return Node(value=mean(labels))

  @classmethod
  def fit(cls, dataset, min_samples_per_split, max_depth):
    return cls.__create_tree(concatenate(dataset, axis=1), min_samples_per_split, max_depth)

def find_best_split(dataset):
  best = {"variance": 0}

  _, features = shape(dataset)
  for feature in range(features - 1):
    values = dataset[:, feature]

    for (threshold, (left, right)) in map(lambda t: (t, perform_split(dataset, feature, t)), unique(values)):
      if len(left) > 0 and len(right) > 0:
        y, left_y, right_y = dataset[:, -1], left[:, -1], right[:, -1]
        variance = math.variance_gain(y, left_y, right_y)

        if variance > best["variance"]:
          best["feature"] = feature
          best["threshold"] = threshold
          best["dataset_left"] = left
          best["dataset_right"] = right
          best["variance"] = variance

  return best

def perform_split(dataset, feature, threshold):
  left = array([row for row in dataset if row[feature] <= threshold])
  right = array([row for row in dataset if row[feature] > threshold])
  return left, right
