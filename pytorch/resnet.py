# -*- coding: utf-8 -*-
# 模型效果：resnet18
#   运行时间：~4h
#   准确率：93% 1epoch
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.datasets import MNIST
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
from torch import nn, optim
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader

class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)


def get_data_loaders(train_batch_size, val_batch_size):
    mnist = MNIST(download=False, train=True, root="./data/MNIST").train_data.float()
    
    data_transform = Compose([ Resize((224, 224)),ToTensor(), Normalize((mnist.mean()/255,), (mnist.std()/255,))])

    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader

start_ts = time.time()

device = torch.device('cpu')
epochs = 1

model = MnistResNet().to(device)
train_loader, val_loader = get_data_loaders(256, 256)

losses = []
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters())

batches = len(train_loader)
val_batches = len(val_loader)

# training loop + eval loop
for epoch in range(epochs):
    total_loss = 0
    model.train()
    
    for i, data in enumerate(train_loader):
        X, y = data[0].to(device), data[1].to(device)
        
        model.zero_grad()
        outputs = model(X)
        loss = loss_function(outputs, y)

        loss.backward()
        optimizer.step()
        current_loss = loss.item()
        total_loss += current_loss
        print('Epoch: ', epoch+1, 'Batch: ', i+1, 'Loss: ', total_loss/(i+1))
    
    val_losses = 0
    precision, recall, f1, accuracy = [], [], [], []
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X, y = data[0].to(device), data[1].to(device)
            outputs = model(X)
            val_losses += loss_function(outputs, y)

            predicted_classes = torch.max(outputs, 1)[1]
            
            total += y.size(0)  # 每一个batch_size 中labels是一个（N，1）的元组，size(0)=N
            correct += (predicted_classes == y).sum().item()  # 对的总个数
    print("Accuracy on the test set %d %%" % (100*correct/total))
        
total_time = time.time() - start_ts
print("Training time: {%d}s" % total_time)
