from matplotlib import pyplot as plt

from dataset import EurovisionDataset

def main():
  dataset = EurovisionDataset.load()

  print(dataset.songs.columns)

if __name__ == '__main__':
  main()
