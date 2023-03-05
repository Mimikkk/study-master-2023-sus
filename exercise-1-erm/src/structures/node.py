from typing import Iterable, overload

def cost(y, y_hat):
  return y != y_hat and 1 or 0

class Node(object):
  def __init__(self):
    self.left = None
    self.right = None
    self.value = 0

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
  def fit(self, data, labels):
    self.perform_split(zip(data, labels))

  @overload
  def predict(self, value: Iterable[float]) -> float: ...
  @overload
  def predict(self, value: Iterable[Iterable[float]]) -> Iterable[float]: ...

  def predict(self, value: Iterable[float] | Iterable[Iterable[float]]) -> int | Iterable[int]:
    value = list(value)
    if isinstance(value[0], Iterable): return map(self.predict, value)
    return value[0] > 0.5 and 1 or 0
