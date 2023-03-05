from typing import Iterable

from numpy import median, concatenate, var, array, mean, unique
import numpy as np
from .node import Node

class DecisionTreeRegressor(object):
  def create_tree(self, dataset, min_samples_per_split, max_depth, depth=0):
    X, Y = dataset[:, :-1], dataset[:, -1]
    samples, features = np.shape(X)

    if samples >= min_samples_per_split and depth <= max_depth:
      best = self.find_best_split(dataset, features)
      if best["variance"] > 0:
        left_subtree = self.create_tree(best["dataset_left"], min_samples_per_split, max_depth, depth + 1)
        right_subtree = self.create_tree(best["dataset_right"], min_samples_per_split, max_depth, depth + 1)
        return Node(
          feature=best["feature"],
          threshold=best["threshold"],
          left=left_subtree,
          right=right_subtree,
          variance=best["variance"]
        )

    return Node(value=mean(Y))

  def find_best_split(self, dataset, num_features):
    best = {}
    max_variance = 0

    for feature in range(num_features):
      features = dataset[:, feature]

      for (threshold, (left, right)) in map(lambda t: (t, perform_split(dataset, feature, t)), unique(features)):
        if len(left) > 0 and len(right) > 0:
          y, left_y, right_y = dataset[:, -1], left[:, -1], right[:, -1]
          variance = variance_gain(y, left_y, right_y)

          if variance > max_variance:
            best["feature"] = feature
            best["threshold"] = threshold
            best["dataset_left"] = left
            best["dataset_right"] = right
            best["variance"] = variance
            max_variance = variance
    return best

  def fit(self, dataset, min_samples_per_split, max_depth):
    return self.create_tree(concatenate(dataset, axis=1), min_samples_per_split, max_depth)

def perform_split(dataset, feature_index, threshold):
  left = array([row for row in dataset if row[feature_index] <= threshold])
  right = array([row for row in dataset if row[feature_index] > threshold])
  return left, right

def variance_gain(common: Iterable[np.ndarray[float]],
                  left: Iterable[np.ndarray[float]],
                  right: Iterable[np.ndarray[float]]):
  left_p = len(left) / len(common)
  right_p = len(right) / len(common)
  return var(common) - left_p * var(left) - right_p * var(right)
