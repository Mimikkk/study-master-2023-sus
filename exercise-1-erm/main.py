from typing import Iterable, overload
class Node(object):
  def __init__(self):
    self.left = None  # Typ: Node, wierzchołek znajdujący się po lewej stornie
    self.right = None  # Typ: Node, wierzchołek znajdujący się po prawej stornie

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

# Najprostsze wczytywanie i zapisywanie danych, a także wywołanie obiektu Node
# Szczególnie przydatne dla osób nie znających numpy
# Zbiór danych jest reprezentowany jako "lista list".
# tj. data[0] zwróci listę z wartościami cech pierwszego przykładu
# Jeśli znasz numpy i bolą Cię od poniższego kodu oczy - możesz go zmienić
# Jeśli nie znasz numpy - skorzystaj z kodu, dokończ zadanie... i naucz sie numpy. W kolejnych tygodniach będziemy z niego korzystać.

# podaj id zbioru danych który chcesz przetworzyć np. 1

class Tree(object):
  @overload
  def predict(self, value: Iterable[float]) -> float: ...
  @overload
  def predict(self, value: Iterable[Iterable[float]]) -> Iterable[float]: ...

  def predict(self, value: Iterable[float] | Iterable[Iterable[float]]) -> int | Iterable[int]:
    value = list(value)
    if isinstance(value[0], Iterable): return map(self.predict, value)
    return value[0] > 0.5 and 1 or 0

def read_csv(filename: str):
  with open(f'resources/database/{filename}.csv', 'r') as csv:
    # skip header
    csv.readline()
    lines = csv.readlines()
  return map(lambda line: map(float, line.strip().split(',')), lines)

def save_csv(filename: str, solution: Iterable[float]):
  header = '"Y"'
  content = map(str, solution)
  with open(filename, 'w') as file: file.writelines(f"{line}\n" for line in (header, *content))

if __name__ == '__main__':
  tree = Tree()

  identifier = "1"
  X = read_csv(f'{identifier}-test')
  Y = tree.predict(X)

  save_csv(filename=f'resources/results/{identifier}.csv', solution=Y)
