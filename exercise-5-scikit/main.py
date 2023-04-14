import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.ensemble
from sklearn import linear_model
from sklearn import neighbors
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import sklearn.tree as trees
import sklearn.ensemble as ensemble


def read():
  columns = np.genfromtxt('resources/datasets/regression.txt', dtype=str, max_rows=1)
  rows = np.genfromtxt('resources/datasets/regression.txt', dtype=np.float32, skip_header=1)
  return pd.DataFrame(data=rows, columns=columns)


def analysis(frame: pd.DataFrame):
  # Analysis
  print(f"Number of attributes: {frame.shape[1] - 1}")
  print(f"Number of rows: {frame.shape[0]}")
  with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(f"columns:")
    print(frame.columns)

    print("Attribute Types:")
    print(frame.dtypes)

    print("Attribute Ranges:")
    print(frame.describe())

  # Box and Violin Plots
  n_columns = frame.shape[1]
  figure, axes = plt.subplots(2, n_columns, figsize=(120, 8))
  for (box_axis, violin_axis), column in zip(zip(*axes), frame.columns):
    sns.boxplot(ax=box_axis, data=frame[[column]])
    box_axis.set_xticklabels(labels=[column])

    sns.violinplot(ax=violin_axis, data=frame[[column]])
    violin_axis.set_xticklabels(labels=[column])

  plt.tight_layout()
  plt.savefig(f'resources/figures/attribute-box-violin-plots.png')
  plt.show()

  # Correlation Matrix
  correlation_matrix = frame.corr()
  figure, axes = plt.subplots(1, 1, figsize=(32, 32))
  sns.heatmap(correlation_matrix, annot=False, ax=axes)
  plt.tight_layout()
  plt.savefig(f'resources/figures/correlation-matrix.png')
  plt.show()


def test_models(frame: pd.DataFrame):
  contents = frame.to_numpy()

  X = contents[:, 0:-1]
  y = contents[:, -1]

  from sklearn.metrics import mean_squared_error

  def calculate_metrics(regressor, X, y):
    return regressor.score(X, y), mean_squared_error(y, regressor.predict(X), squared=False)

  def format_scores(title, r2, rmse):
    return f"{title: >20}:\n\tR2: {r2:.2f} | RMSE: {rmse:.2f}"

  scores = []
  for (name, regressor) in [
    ("Random Forest", RandomForestRegressor()),
    ("Linear Regression", linear_model.LinearRegression()),
    ("Ridge", linear_model.Ridge()),
    ("Lasso", linear_model.Lasso()),
    ("Elastic Net", linear_model.ElasticNet()),
    ("Bayesian Ridge", linear_model.BayesianRidge()),
    ("Knn-1", neighbors.KNeighborsRegressor(n_neighbors=1)),
    ("Knn-2", neighbors.KNeighborsRegressor(n_neighbors=2)),
    ("Knn-3", neighbors.KNeighborsRegressor(n_neighbors=3)),
    ("Knn-4", neighbors.KNeighborsRegressor(n_neighbors=4)),
    ("Knn-8", neighbors.KNeighborsRegressor(n_neighbors=8)),
    ("Knn-13", neighbors.KNeighborsRegressor(n_neighbors=13)),
    ("Knn-21", neighbors.KNeighborsRegressor(n_neighbors=21)),
    ("Tree-2", DecisionTreeRegressor(max_depth=2)),
    ("Tree-5", DecisionTreeRegressor(max_depth=5)),
    ("Tree-8", DecisionTreeRegressor(max_depth=8)),
    ("Tree-unbound", DecisionTreeRegressor(max_depth=None)),
    ("Pony", MLPRegressor()),
    ("svr-rbf", SVR(kernel="rbf")),
    ("svr-linear", SVR(kernel="linear")),
    ("svr-poly", SVR(kernel="poly")),
    ("ard", linear_model.ARDRegression()),
    ("sgd", linear_model.SGDRegressor()),
    ("adaboost", ensemble.AdaBoostRegressor()),
    ("gradientboosting", ensemble.GradientBoostingRegressor()),
  ]:
    regressor.fit(X, y)
    scores.append((name, calculate_metrics(regressor, X, y)))

  for name, (r2, rmse) in scores: print(format_scores(name, r2, rmse))

  r2_scores = [r2 for _, (r2, _) in scores]
  rmse_scores = [rmse for _, (_, rmse) in scores]
  figure, axes = plt.subplots(1, 2, figsize=(32, 8))
  sns.barplot(x=[name for name, _ in scores], y=r2_scores, ax=axes[0])
  axes[0].set_title("R2 Scores")
  axes[0].set_ylabel("R2 Score")
  axes[0].set_xticklabels(labels=[name for name, _ in scores], rotation=60)

  sns.barplot(x=[name for name, _ in scores], y=rmse_scores, ax=axes[1])
  axes[1].set_title("RMSE Scores")
  axes[1].set_ylabel("RMSE Score")
  axes[1].set_xticklabels(labels=[name for name, _ in scores], rotation=60)

  plt.tight_layout()
  plt.savefig(f'resources/figures/model-scores.png')
  plt.show()


def main():
  frame = read()

  # analysis(frame)

  test_models(frame)


if __name__ == '__main__':
  main()
