import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn import neighbors
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import sklearn.ensemble as ensemble
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler


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


def calculate_metrics(regressor, X, y):
  return regressor.score(X, y), mean_squared_error(y, regressor.predict(X), squared=False)


def format_scores(title, r2, rmse):
  return f"{title: >20}:\n\tR2: {r2:.2f} | RMSE: {rmse:.2f}"


def test_models(frame: pd.DataFrame):
  contents = frame.to_numpy()

  X = contents[:, 0:-1]
  y = contents[:, -1]

  scores = []
  for (name, regressor) in [
    ("Random Forest-2", RandomForestRegressor(max_depth=2)),
    ("Random Forest-5", RandomForestRegressor(max_depth=5)),
    ("Random Forest-8", RandomForestRegressor(max_depth=8)),
    ("Random Forest-unbound", RandomForestRegressor(max_depth=None)),
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
  plt.savefig(f'resources/figures/normalized-model-metrics.png')
  plt.show()


def test_models_with_normalized_features(frame: pd.DataFrame):
  contents = frame.to_numpy()

  X = contents[:, 0:-1]
  y = contents[:, -1]

  scaler = RobustScaler()
  X = scaler.fit_transform(X)

  scores = []
  for (name, regressor) in [
    ("Random Forest-2", RandomForestRegressor(max_depth=2)),
    ("Random Forest-5", RandomForestRegressor(max_depth=5)),
    ("Random Forest-8", RandomForestRegressor(max_depth=8)),
    ("Random Forest-unbound", RandomForestRegressor(max_depth=None)),
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
  axes[0].set_title("R2 Scores with normalized features")
  axes[0].set_ylabel("R2 Score")
  axes[0].set_xticklabels(labels=[name for name, _ in scores], rotation=60)

  sns.barplot(x=[name for name, _ in scores], y=rmse_scores, ax=axes[1])
  axes[1].set_title("RMSE Scores with normalized features")
  axes[1].set_ylabel("RMSE Score")
  axes[1].set_xticklabels(labels=[name for name, _ in scores], rotation=60)

  plt.tight_layout()
  plt.savefig(f'resources/figures/normalized-model-metrics.png')
  plt.show()


def present_supposedly_best_model(frame: pd.DataFrame):
  contents = frame.to_numpy()

  X = contents[:, 0:-1]
  y = contents[:, -1]

  scaler = RobustScaler()
  X = scaler.fit_transform(X)

  model = DecisionTreeRegressor(max_depth=5)
  model.fit(X, y)

  plt.figure(figsize=(60, 30))
  plot_tree(model, fontsize=8, filled=True, rounded=True, precision=1)
  plt.savefig(f'resources/figures/best-model-tree.png', dpi=100)
  plt.show()

  y_pred = model.predict(X)

  plt.scatter(y, y_pred, marker='o', s=3, alpha=0.9)
  plt.xlabel('Faktyczne wartości')
  plt.ylabel('Przewidywane wartości')
  plt.title('DecisionTreeRegressor(max_depth=5)')

  plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k', lw=1, alpha=0.8)
  plt.tight_layout()

  plt.savefig(f'resources/figures/best-model-slice.png')
  plt.show()


def validate_models_with_normalized_features_using_10folds(frame: pd.DataFrame):
  contents = frame.to_numpy()

  X = contents[:, 0:-1]
  y = contents[:, -1]

  scaler = RobustScaler()
  X = scaler.fit_transform(X)

  scores = []
  k_fold = KFold(n_splits=10, shuffle=True)
  for (name, regressor) in [
    ("Random Forest-2", RandomForestRegressor(max_depth=2)),
    ("Random Forest-5", RandomForestRegressor(max_depth=5)),
    ("Random Forest-8", RandomForestRegressor(max_depth=8)),
    ("Random Forest-unbound", RandomForestRegressor(max_depth=None)),
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
    print(f"Validating {name}...")
    rmse_scores = -cross_val_score(regressor, X, y, cv=k_fold, scoring='neg_root_mean_squared_error')
    r2_scores = cross_val_score(regressor, X, y, cv=k_fold, scoring='r2')
    rmse_avg = np.average(rmse_scores)
    r2_avg = np.average(r2_scores)
    scores.append((name, (r2_avg, rmse_avg)))

  for name, (r2, rmse) in scores: print(format_scores(name, r2, rmse))

  r2_scores = [r2 for _, (r2, _) in scores]
  rmse_scores = [rmse for _, (_, rmse) in scores]
  figure, axes = plt.subplots(1, 2, figsize=(32, 8))
  sns.barplot(x=[name for name, _ in scores], y=r2_scores, ax=axes[0])
  axes[0].set_title("R2 Average Scores with normalized features")
  axes[0].set_ylabel("R2 Average Score")
  axes[0].set_xticklabels(labels=[name for name, _ in scores], rotation=60)

  sns.barplot(x=[name for name, _ in scores], y=rmse_scores, ax=axes[1])
  axes[1].set_title("RMSE Average Scores with normalized features")
  axes[1].set_ylabel("RMSE Average Score")
  axes[1].set_xticklabels(labels=[name for name, _ in scores], rotation=60)

  plt.tight_layout()
  plt.savefig(f'resources/figures/10fold-model-metrics.png')
  plt.show()


def prelude():
  import os
  if not os.path.exists('resources/figures'):
    os.makedirs('resources/figures')

  if not os.path.exists('resources/datasets'):
    os.makedirs('resources/datasets')

  if not os.path.exists('resources/datasets/regression.txt'):
    raise Exception("Missing dataset file: resources/datasets/regression.txt")


def main():
  prelude()

  frame = read()

  analysis(frame)

  test_models(frame)

  test_models_with_normalized_features(frame)

  present_supposedly_best_model(frame)

  validate_models_with_normalized_features_using_10folds(frame)


if __name__ == '__main__':
  main()
