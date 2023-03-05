from typing import Iterable

def read(filename: str):
  with open(f'resources/database/{filename}.csv', 'r') as csv:
    # skip header
    csv.readline()
    lines = csv.readlines()
  return map(lambda line: map(float, line.strip().split(',')), lines)
def save(filename: str, solution: Iterable[float]):
  header = '"Y"'
  content = map(str, solution)
  with open(filename, 'w') as file: file.writelines(f"{line}\n" for line in (header, *content))
