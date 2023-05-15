from imblearn.metrics import geometric_mean_score
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
    KNeighborsClassifier(),
    DecisionTreeClassifier(class_weight='balanced'),
    RandomForestClassifier(class_weight='balanced'),
    SVC(class_weight='balanced'),
    MLPClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
  ]
  cv = StratifiedKFold(n_splits=10)

  results = []
  for classifier in classifiers:
    print(f'Classifier: {classifier.__class__.__name__}')
    resampler = SMOTE()
    pipeline = Pipeline([
      ('SMOTE', resampler),
      ('Standardization', StandardScaler()),
      ('Classifier', classifier)
    ])

    print(f'Accuracy')
    accuracy = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv).mean()
    print(f'Roc AUC')
    roc_auc = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv).mean()
    print(f'G-mean')
    g_mean = cross_val_score(pipeline, X, y, scoring=gmean(), cv=cv).mean()
    results.append((accuracy, roc_auc, g_mean))

  labels = ['KNN', 'D-Tree', 'R-Forest', 'SVC', 'MLP', 'GaussianNB', 'QDA']
  accuracy, roc_auc, g_mean = zip(*results)

  x = np.arange(len(labels))
  width = 0.2

  fig, ax = plt.subplots()
  ax.bar(x - width, accuracy, width, label='Accuracy')
  ax.bar(x, roc_auc, width, label='ROC AUC')
  ax.bar(x + width, g_mean, width, label='G-Mean')
  ax.set_ylabel('Scores')
  ax.set_title('Scores by classifier and metric (znormalizowane)')
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.legend()

  fig.tight_layout()
  plt.show()

if __name__ == '__main__':
  main()
