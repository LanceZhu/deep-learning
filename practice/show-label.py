import pickle
import numpy as np
import gzip

with gzip.open('../data/MNIST/train-labels-idx1-ubyte.gz', 'rb') as file:
  data = np.frombuffer(file.read(), np.uint8, offset=8)
  data = data.reshape(-1, 1)
  print(data[1])
