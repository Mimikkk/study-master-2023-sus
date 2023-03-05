import argparse
import os

def construct_solution(directory: str):
  solution = []
  for i in range(1, 14):
    line = [line.strip() for line in open(os.path.join(directory, f"{i}.csv"))]
    if line[0].strip("\"") in ("Y", "x"): del line[0]

    if i == 8: line.extend(line)
    elif i == 3:
      line.extend(line)
      line.extend(line)

    assert len(line) == 200
    solution.extend(line)
  return solution

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="directory with solution files")
parser.add_argument("resultname", help="name of the output file")

if __name__ == '__main__':
  args = parser.parse_args()
  solution = construct_solution(args.directory)

  with open(args.resultname, 'w') as file:
    file.write("id,Y\n")
    for i, line in enumerate(solution): file.write(f"{i},{line}\n")
