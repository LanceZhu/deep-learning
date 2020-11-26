# -*- coding: utf-8 -*-
# 模型效果
#   耗时 ~6min
#   准确率 98%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Cnn(nn.Module):
  def __init__(self):
    super(Cnn, self).__init__()
    self.conv = nn.Conv2d(1, 32, 3, 1)
    self.fc = nn.Linear(32*26*26, 10)
  
  def forward(self, x):
    x = self.conv(x)
    x = F.relu(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x

model = Cnn()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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
                         batch_size=64,
                         shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=64,
                         shuffle=False)

def train(epoch):
    runing_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()
        
        runing_loss += loss.item()
        if batch_idx % 300  == 299:
            print("[%d, %5d] loss: %.3f" % (epoch+1, batch_idx+1, runing_loss/300))
            runing_loss = 0.0
  
def test():
    correct = 0
    total =0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)  
            _, predicted = torch.max(outputs.data, dim=1 )  # 返回两个值，第一个是最大值，第二个是最大值的索引。dim=1表示在列维度求以上结果，dim = 0表示在行维度求以上结果。         
            total += labels.size(0)  # 每一个batch_size 中labels是一个（N，1）的元组，size(0)=N
            correct += (predicted == labels).sum().item()  # 对的总个数
    print("Accuracy on the test set %d %%" % (100*correct/total))

if __name__=="__main__":
    for epoch in range(10):
        train(epoch)
        test()
