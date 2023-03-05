import numpy as np
from .node import Node
from .dataset import DataSet


class DecisionTreeRegressor(object):
  def __init__(self, min_samples_split=2, max_depth=2):
    self.root = None

    self.min_samples = min_samples_split
    self.max_depth = max_depth

  def build_tree(self, dataset, depth=0):
    x, y = dataset[:, :-1], dataset[:, -1]
    (samples, features) = np.shape(x)

    if samples >= self.min_samples and depth <= self.max_depth:
      best = calculate_best_split(dataset, features)
      if best["variance"] > 0:
        left_subtree = self.build_tree(best["dataset_left"], depth + 1)
        right_subtree = self.build_tree(best["dataset_right"], depth + 1)
        return Node(best["feature"], best["threshold"], left_subtree, right_subtree, best["variance"])

    return Node(value=np.mean(y))

  def print_node(self, tree=None, indent=" "):
    if not tree: tree = self.root

    if tree.value is not None:
      print(tree.value)

    else:
      print(f"X_{str(tree.feature)} <= {tree.threshold} ? {tree.variance}")
      print(f"{indent}left:", end="")
      self.print_node(tree.left, indent + indent)
      print(f"{indent}right:", end="")
      self.print_node(tree.right, indent + indent)

  def fit(self, dataset: DataSet):
    self.root = self.build_tree(np.concatenate(dataset, axis=1))

  def _predict(self, tree: Node, value):
    if tree.value is not None: return tree.value

    value = value[tree.feature]
    if value <= tree.threshold:
      return self._predict(tree.left, value)
    else:
      return self._predict(tree.right, value)

  def predict(self, X):
    return [self._predict(self.root, x) for x in X]

def calculate_variance(parent, l_child, r_child):
  left = len(l_child) / len(parent)
  right = len(r_child) / len(parent)
  return np.var(parent) - left * np.var(l_child) - right * np.var(r_child)

def calculate_best_split(dataset, feature_count):
  best = {}
  best_variance = -float("inf")

  for feature in range(feature_count):
    thresholds = np.unique(dataset[:, feature])
    # loop over all the feature values present in the data
    for threshold in thresholds:
      left, right = split_ds(dataset, feature, threshold)

      # check if childs are not null
      if len(left) > 0 and len(right) > 0:
        y, left_y, right_y = dataset[:, -1], left[:, -1], right[:, -1]
        variance = calculate_variance(y, left_y, right_y)

        if variance > best_variance:
          best["feature"] = feature
          best["threshold"] = threshold
          best["dataset_left"] = left
          best["dataset_right"] = right
          best["variance"] = variance
          best_variance = variance
  return best

def split_ds(dataset, feature: int, threshold: float):
  left = np.array([row for row in dataset if row[feature] <= threshold])
  right = np.array([row for row in dataset if row[feature] > threshold])
  return left, right
