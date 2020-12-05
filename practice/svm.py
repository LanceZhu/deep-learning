# -*- coding: utf-8 -*-
# 模型效果（前5000数据）
#   耗时：~2min
#   准确率：93.5%
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# import pickle

# iris = load_iris()
# x = iris.data
# y = iris.target

transform = transforms.Compose([
    transforms.ToTensor(),  # 转张量，将值缩放到[0,1]之间
    transforms.Normalize((0.1307,),(0.3081,))  # 归一化，第一个为均值，第二个为方差
])
train_dataset = datasets.MNIST(root= "./data/MNIST",
                              train=True,  # 下载训练集
                              transform=transform,  # 转张量，将值缩放到[0,1]之间.也可以写成transform = transforms.ToTensor()      
                              download=True
                              )

test_dataset = datasets.MNIST(root= "./data/MNIST",
                              train=False,  # 下载训练集
                              transform=transform,  # 转张量，将值缩放到[0,1]之间
                              download=True
                              )
train_loader = DataLoader(dataset=train_dataset,
                         shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         shuffle=False)

train_image_arr = []
train_label_arr = []
for batch_idx, data in enumerate(train_loader, 0):
  if (batch_idx < 5000):
    inputs, target = data
    train_image_arr.append(inputs.view(-1, 28*28).numpy()[0])
    train_label_arr.append(target.numpy()[0])
  else:
    break

test_image_arr = []
test_label_arr = []
for batch_idx, data in enumerate(test_loader, 0):
  if (batch_idx < 5000):
    inputs, target = data
    test_image_arr.append(inputs.view(-1, 28*28).numpy()[0])
    test_label_arr.append(target.numpy()[0])
  else:
    break

svc = SVC(gamma='auto')
svc.fit(train_image_arr, train_label_arr)

print(svc.predict(test_image_arr))

print(accuracy_score(svc.predict(test_image_arr), test_label_arr))

# output = open('svm.pkl', 'wb')
# pickle.dump(svc, output)
# output.close()
