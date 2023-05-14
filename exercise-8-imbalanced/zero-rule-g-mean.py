from numpy import mean
from numpy import std
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from matplotlib import pyplot
import dataset as ds

def evaluate_model(X, y, model):
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
  metric = make_scorer(geometric_mean_score)
  scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
  return scores

def get_models():
  models, names = list(), list()
  models.append(DummyClassifier(strategy='most_frequent'))
  names.append('Zero-Rule | Majority')
  return models, names

def main():
  dataset: DataFrame = ds.load()
  X, y = dataset.values[:, :-1], dataset.values[:, -1]

  models, names = get_models()
  results = list()
  for i in range(len(models)):
    scores = evaluate_model(X, y, models[i])
    results.append(scores)
    print(f'>{names[i]} {mean(scores):.3f} ({std(scores):.3f})')
  pyplot.boxplot(results, labels=names, showmeans=True)
  pyplot.show()

if __name__ == '__main__':
  main()
