from dataset import EurovisionDataset

def main():
  dataset = EurovisionDataset.load()
  print(dataset.votes.year2016)
  print(dataset.votes.year2016.columns)

if __name__ == '__main__':
  main()
