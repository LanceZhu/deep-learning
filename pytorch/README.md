# MNIST 手写数字识别实践 - PyTorch + sklearn

## 基本介绍

:link: [MNIST 手写数字识别实践](http://hexo.f00bar.top/2020/12/17/handwriting-number-recognition-mnist/)

## 具体模型

1·. PyTorch

- DNN（深度神经网络）

  具体实现：

  ​	:page_facing_up: ​[./dnn.py](./dnn.py)

  网络结构：

  ​	增加隐藏层，激活函数为ReLU，优化器SGD。

- CNN（卷积神经网络）

  具体代码：

  ​	:page_facing_up: [./cnn.py](./cnn.py)

  网络结构：

  ​	增加卷积层，激活函数为ReLu，优化器SGD。

2. sklearn

- KNN（K近邻算法）

  具体实现：

  ​	:page_facing_up: [./knn.py](./knn.py)

- SVM（支持向量机）

  具体实现：

  ​	:page_facing_up: [./svm.py](./svm.py)​