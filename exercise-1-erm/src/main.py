import utils.csv as csv
from structures import Node

if __name__ == '__main__':
  identifier = "1"

  tree = Node()
  x = csv.read(f'{identifier}-X')
  Y = csv.read(f'{identifier}-Y')
  tree.fit(x, Y)

  x = csv.read(f'{identifier}-test')
  Y = tree.predict(x)

  csv.save(filename=f'resources/results/{identifier}.csv', solution=Y)
