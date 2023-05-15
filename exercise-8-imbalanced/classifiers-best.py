from pandas import DataFrame
import matplotlib.pyplot as plt
import dataset as ds
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.datasets import make_classification
from imblearn.metrics import geometric_mean_score
import seaborn as sns
import pandas as pd

def gmean():
  return make_scorer(geometric_mean_score)

def main():
  df: DataFrame = ds.load()
  X, y = df.values[:, :-1], df.values[:, -1]

  param_grid = {
    'n_estimators': [200],
    'max_depth': [10],
    'min_samples_split': [10],
    'min_samples_leaf': [4],
  }

  rfc = RandomForestClassifier(random_state=42, class_weight='balanced')

  gcv = GridSearchCV(rfc, param_grid, scoring=gmean(), cv=10, n_jobs=-1)

  gcv.fit(X, y)

  cv_results = pd.DataFrame(gcv.cv_results_)

  cv_results_melted = cv_results.melt(
    id_vars=['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf'],
    value_vars=['std_test_score'],
    var_name='Metric',
    value_name='Score')

  plt.figure(figsize=(10, 6))
  sns.heatmap(cv_results_melted.pivot(index='param_n_estimators', columns='param_max_depth', values='Score'),
              annot=True, fmt=".3f", linewidths=.5, cmap='Blues')
  plt.title('Wyniki przeszukiwania')
  plt.show()

if __name__ == '__main__':
  main()
