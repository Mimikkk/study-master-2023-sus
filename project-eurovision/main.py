from dataset import EurovisionDataset

def main():
  dataset = EurovisionDataset.load()
  print(dataset.votes._2016)

if __name__ == '__main__':
  main()
