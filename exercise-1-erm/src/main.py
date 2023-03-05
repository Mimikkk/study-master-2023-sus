import utils.csv as csv
from structures import Node

if __name__ == '__main__':
  tree = Node()

  identifier = "1"
  X = csv.read(f'{identifier}-test')
  Y = tree.predict(X)

  csv.save(filename=f'resources/results/{identifier}.csv', solution=Y)
