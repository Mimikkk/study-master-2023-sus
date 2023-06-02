from matplotlib import pyplot as plt

from dataset import EurovisionDataset

def main():
  dataset = EurovisionDataset.load()

  print(dataset.songs.in_native.value_counts())
  print(dataset.songs.in_english.value_counts())


if __name__ == '__main__':
  main()
