from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
import os.path as path

data_dir = '../data/test/naruto'

image = Image.open(path.join(data_dir, 'naruto.png'))

img_gray = np.array(image.convert("L"), dtype=np.float32)

# gray image
plt.imsave(path.join(data_dir, 'naruto-gray.png'), img_gray)

img_gray_tensor = torch.Tensor(img_gray)
img_gray_tensor = img_gray_tensor.view(-1, 1, 543, 543)

# image conv 
kernel_size = 3
kernel = torch.Tensor([
  [-1, -1, 0],
  [-1, 0, 1],
  [0, 1, 1]
])
kernel = kernel.reshape((1, 1, kernel_size, kernel_size))

conv2d = nn.Conv2d(1, 1, kernel_size, bias=False)
conv2d.weight.data[0] = kernel

img_conv = conv2d(img_gray_tensor)

plt.imsave(path.join(data_dir, 'naruto-conv.png'), img_conv)