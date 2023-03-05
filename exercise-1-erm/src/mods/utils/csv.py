from typing import Iterable
import numpy as np
from ..fp import exhaust

def read(filename: str) -> np.ndarray[float]:
  with open(f'resources/database/{filename}.csv', 'r') as csv:
    # skip header
    csv.readline()
    lines = csv.readlines()
  return np.array(exhaust(map(lambda line: map(float, line.strip().split(',')), lines)))

def save(filename: str, solution: Iterable[float]):
  header = '"Y"'
  content = map(str, solution)
  with open(filename, 'w') as file: file.writelines(f"{line}\n" for line in (header, *content))
