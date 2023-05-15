from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from pandas import DataFrame
import matplotlib.pyplot as plt
import dataset as ds
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, confusion_matrix
from imblearn.metrics import geometric_mean_score
import seaborn as sns
import pandas as pd

def main():
  df: DataFrame = ds.load()
  X, y = df.values[:, :-1], df.values[:, -1]

  weight_ratios = [{
    0: 0.1,
    1: 0.9
  }, {
    0: 0.2,
    1: 0.8
  }, {
    0: 1,
    1: 1
  }, "balanced_subsample", 'balanced']
  false_positives = []
  false_negatives = []

  for weight_ratio in weight_ratios:
    pipeline = Pipeline([
      ('smote', SMOTE(random_state=42)),
      ('scaler', StandardScaler()),
      ('clf', RandomForestClassifier(
        random_state=42,
        class_weight=weight_ratio,
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4
      ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    false_positives.append(fp)
    false_negatives.append(fn)

  plt.figure(figsize=(8, 6))
  plt.plot(weight_ratios, false_positives, label='False Positives')
  plt.plot(weight_ratios, false_negatives, label='False Negatives')
  plt.xlabel('Proporcja wag')
  plt.ylabel('Liczba')
  plt.title('Liczba FP i FN w zależności od proporcji wag')
  plt.legend()
  plt.show()

if __name__ == '__main__':
  main()
