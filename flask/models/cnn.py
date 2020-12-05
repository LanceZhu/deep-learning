# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import transforms

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
model.load_state_dict(torch.load('models/state/cnn'))
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
