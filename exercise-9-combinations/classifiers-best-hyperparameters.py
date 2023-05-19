from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from pandas import DataFrame
import matplotlib.pyplot as plt
import dataset as ds
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score
import seaborn as sns
import pandas as pd

def gmean():
  return make_scorer(geometric_mean_score)

def main():
  df: DataFrame = ds.load()
  X, y = df.values[:, :-1], df.values[:, -1]

  param_grid = {
    'clf__class_weight': ['balanced', None],
    'clf__n_estimators': [10, 50, 100],
    'clf__max_depth': [2, 5, 8],
    'clf__min_samples_split': [1, 8, 16, 32, 64],
    'clf__min_samples_leaf': [1, 8, 16, 32, 64],
  }

  pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
  ])

  gcv = GridSearchCV(pipeline, param_grid, scoring={
    'gmean': gmean(),
    'accuracy': 'accuracy',
  }, cv=5, n_jobs=-1, verbose=1)

  gcv.fit(X, y)

  cv_results = pd.DataFrame(gcv.cv_results_)

  # Select the parameters and score columns
  subset = [
    'param_clf__class_weight',
    'param_clf__n_estimators',
    'param_clf__max_depth',
    'param_clf__min_samples_split',
    'param_clf__min_samples_leaf',
    'mean_test_score']

  cv_results_sub = cv_results[subset]

  cv_results_sub['params_combined'] = cv_results_sub[subset[:5]].apply(lambda row: '_'.join(row.values.astype(str)),
                                                                       axis=1)

  cv_results_melted = cv_results_sub.melt(id_vars='params_combined', value_vars='mean_test_score', var_name='Metric',
                                          value_name='Score')

  pivot = cv_results_melted.pivot(index='params_combined', columns='Metric', values='Score')

  plt.figure(figsize=(10, 120))
  sns.heatmap(pivot, annot=True, fmt=".3f", linewidths=.5, cmap='Blues')
  plt.title('Wyniki')
  plt.xlabel('G-mean')
  plt.ylabel(f'Konfiguracja (class_weight, n_estimators, max_depth, min_samples_split, min_samples_leaf)')
  plt.show()


if __name__ == '__main__':
  main()
