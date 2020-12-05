# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class Dnn(nn.Module):
  def __init__(self):
    super(Dnn, self).__init__()
    self.fc1 = nn.Linear(28*28, 64)
    self.fc2 = nn.Linear(64, 10)
  
  def forward(self, x):
    x = x.view(-1, 28*28)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    return x

model = Dnn()
model.load_state_dict(torch.load('models/state/dnn'))
model.eval()

def transfrom_image(image):
  transform = transforms.Compose([
    # transforms.Resize((28, 28)),
    # transforms.CenterCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
  ])
  return transform(image).unsqueeze(0)

def predict(image):
  tensor = transfrom_image(image)
  result = model(tensor)
  _, num_tensor = torch.max(result.data, dim=1)
  num = num_tensor.item()
  return num
