from typing import Iterable

from numpy import median, concatenate, var, array
import numpy as np
from .node import Node

class DecisionTreeRegressor(object):
  def create_tree(self, dataset, min_samples_per_split, max_depth, depth=0):
    X, Y = dataset[:, :-1], dataset[:, -1]
    num_samples, num_features = np.shape(X)

    if num_samples >= min_samples_per_split and depth <= max_depth:
      best = self.find_best_split(dataset, num_features)
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

    return Node(value=median(Y))

  def find_best_split(self, dataset, num_features):
    best = {}
    max_variance = -float("inf")

    for feature in range(num_features):
      features = dataset[:, feature]
      candidate_thresholds = np.unique(features)

      for threshold in candidate_thresholds:
        left, right = self.perform_split(dataset, feature, threshold)
        if len(left) > 0 and len(right) > 0:
          y, left_y, right_y = dataset[:, -1], left[:, -1], right[:, -1]
          variance = self.variance(y, left_y, right_y)

          if variance > max_variance:
            best["feature"] = feature
            best["threshold"] = threshold
            best["dataset_left"] = left
            best["dataset_right"] = right
            best["variance"] = variance
            max_variance = variance
    return best

  def perform_split(self, dataset, feature_index, threshold):
    left = array([row for row in dataset if row[feature_index] <= threshold])
    right = array([row for row in dataset if row[feature_index] > threshold])
    return left, right

  def variance(self, parent, l_child, r_child):
    left = len(l_child) / len(parent)
    right = len(r_child) / len(parent)
    return var(parent) - left * var(l_child) - right * var(r_child)

  def fit(self, dataset, min_samples_per_split, max_depth):
    return self.create_tree(concatenate(dataset, axis=1), min_samples_per_split, max_depth)
