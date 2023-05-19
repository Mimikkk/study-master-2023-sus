from imblearn.metrics import geometric_mean_score
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import dataset as ds

def gmean():
  return make_scorer(geometric_mean_score)

def main():
  df: DataFrame = ds.load()
  X, y = df.values[:, :-1], df.values[:, -1]

  classifiers = [
    AdaBoostClassifier(estimator=DecisionTreeClassifier()),
    RandomForestClassifier(),
    VotingClassifier(estimators=[
      ('dt', DecisionTreeClassifier()),
      ('svc', SVC()),
      ('mlp', MLPClassifier())
    ]),
    StackingClassifier(estimators=[
      ('dt', DecisionTreeClassifier()),
      ('svc', SVC()),
      ('mlp', MLPClassifier())
    ]),
  ]
  cv = StratifiedKFold(n_splits=10)

  metrics = []
  for classifier in classifiers:
    print(f'Classifier: {classifier.__class__.__name__}')
    standard_scaler = StandardScaler()
    pipeline = Pipeline([
      ('Scaler', standard_scaler),
      ('Classifier', classifier)
    ])

    print(f'Accuracy')
    accuracy = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv).mean()
    print(f'G-mean')
    gmean_score = cross_val_score(pipeline, X, y, scoring=gmean(), cv=cv).mean()
    metrics.append((accuracy, gmean_score))

  labels = [
    'AdaBoost-DT',
    'R-Forest',
    'Voting-1',
    'Stacking-1',
  ]

  x = np.arange(len(labels))
  width = 0.2

  accuracy, g_mean = zip(*metrics)
  fig, ax = plt.subplots()
  ax.bar(x - width, accuracy, width, label='Trafność')
  ax.bar(x + width, g_mean, width, label='G-Mean')
  ax.set_ylabel('Wyniki')
  ax.set_title('Wynik względem klasyfikatora i metryki (znormalizowane)')
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.legend()

  fig.tight_layout()
  plt.show()

if __name__ == '__main__':
  main()
