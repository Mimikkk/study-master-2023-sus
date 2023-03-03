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

id = "1"  # podaj id zbioru danych który chcesz przetworzyć np. 1
data = []
y = [line.strip() for line in open(id + '-Y.csv')]
for i, line in enumerate(open(id + '-X.csv')):
  if i == 0: continue
  x = [float(j) for j in line.strip().split(',')]
  nAttr = len(x)
  x.append(float(y[i]))
  data.append(x)
print('Data load complete!')
tree = Node()
tree.perform_split(data)
print('Training complete!')

with open(id + '.csv', 'w') as f:
  for i, line in enumerate(open(id + '-test.csv')):
    if i == 0:
      continue
    x = [float(j) for j in line.strip().split(',')]
    y = tree.predict(x)
    f.write(str(y))
    f.write('\n')
