from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt


def main():
  point_count = 40
  neuron_count = 10
  t = np.linspace(0, np.pi * 2, point_count)
  x = t
  y = np.sin(t)

  som = MiniSom(1, neuron_count, 2, sigma=0.2, learning_rate=0.1, neighborhood_function='gaussian', random_seed=0)
  points = np.array([x, y]).T
  som.random_weights_init(points)

  plt.figure(figsize=(10, 9))
  total_iter = 0

  # note: increasing training periods
  for (i, iterations) in enumerate(range(5, 116, 10)):
    som.train(points, iterations, verbose=False, random_order=False)
    total_iter += iterations
    plt.subplot(3, 4, i + 1)
    plt.scatter(x, y, color='red', s=10)

    plt.plot(som.get_weights()[0][:, 0], som.get_weights()[0][:, 1], 'green', marker='o')
    plt.title(f"Iterations: {total_iter:d}\nError: {som.quantization_error(points):.3f}")
    plt.xticks([])
    plt.yticks([])
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  main()
