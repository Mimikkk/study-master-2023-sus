import numpy as np

def main():
  headers = np.genfromtxt('resources/regression.txt', dtype=str, max_rows=1)
  rows = np.genfromtxt('resources/regression.txt', dtype=float, skip_header=1)

  print(headers)
  print(rows)


if __name__ == '__main__':
    main()
