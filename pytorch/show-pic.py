import pickle
from matplotlib import pyplot
import numpy as np
import gzip

with gzip.open('./data/MNIST/train-images-idx3-ubyte.gz', 'rb') as file:
  data = np.frombuffer(file.read(), np.uint8, offset=16)
  data = data.reshape(-1, 784)
  # print(data[1])
  pyplot.imsave('data/mnist-example-1.png', data[1].reshape((28, 28)), cmap='gray')
