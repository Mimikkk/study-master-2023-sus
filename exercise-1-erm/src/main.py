import utils.csv as csv

from typing import Iterable, overload
class Node(object):
  def __init__(self):
    self.left = None
    self.right = None

  def perform_split(self, data):
    ...
  # Znajdź najlepszy podział data
  # if uzyskano poprawę funkcji celu (bądź inny, zaproponowany przez Ciebie warunek):
  # podziel dane na dwie części d1 i d2, zgodnie z warunkiem
  # self.left = Node()
  # self.right = Node()
  # self.left.perform_split(d1)
  # self.right.perform_split(d2)
  # else:
  # obecny Node jest liściem, zapisz jego odpowiedź

  def predict(self, example):
    ...
    """
    if not Node jest liściem:
      if warunek podziału jest spełniony:
        return self.right.predict(example)
      else:
        return self.left.predict(example)
    return zwróć wartość (Node jest liściem)
    """

class Tree(object):
  @overload
  def predict(self, value: Iterable[float]) -> float: ...
  @overload
  def predict(self, value: Iterable[Iterable[float]]) -> Iterable[float]: ...

  def predict(self, value: Iterable[float] | Iterable[Iterable[float]]) -> int | Iterable[int]:
    value = list(value)
    if isinstance(value[0], Iterable): return map(self.predict, value)
    return value[0] > 0.5 and 1 or 0

if __name__ == '__main__':
  tree = Tree()

  identifier = "1"
  X = csv.read(f'{identifier}-test')
  Y = tree.predict(X)

  csv.save(filename=f'resources/results/{identifier}.csv', solution=Y)
