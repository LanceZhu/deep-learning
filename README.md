# deep-learning

deep learning 学习/实践

## 框架/库

- PyTorch - https://pytorch.org/
- TensorFlow - https://tensorflow.google.cn/

## 实践

### MNIST（手写数字识别）

数据集：http://yann.lecun.com/exdb/mnist/

#### PyTorch 实现

官方示例 examples/mnist - https://github.com/pytorch/examples/tree/master/mnist

```bash
$ git clone https://github.com/pytorch/examples.git
$ cd examples/mnist
$ pip install -r requirements.txt
$ python main.py
```

##### 模型效果

| 指标         | 内容   |
| ------------ | ------ |
| 训练耗时     | ~40min |
| 内存占用     | ~50%   |
| 准确率       | 99%    |
| Average loss | 0.0266 |

##### 软硬件信息

| 指标 | 内容              |
| ---- | ----------------- |
| OS   | Windows10 中 wsl  |
| CPU  | Core(TM) i7-6500U |
| RAM  | 12GB              |
