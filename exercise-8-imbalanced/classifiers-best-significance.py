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

  pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
      random_state=42, class_weight='balanced',
      n_estimators=200,
      max_depth=10,
      min_samples_split=10,
      min_samples_leaf=4
    ))
  ])

  pipeline.fit(X, y)

  importance = pipeline.steps[2][1].feature_importances_

  feature_importances = pd.Series(importance, df.columns[:-1])

  feature_importances.sort_values(ascending=False, inplace=True)

  feature_importances.plot(x='Cechy', y='Znaczenie', kind='bar', figsize=(60, 9), rot=60, fontsize=15)

  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  main()
