# -*- coding: utf-8 -*-
import pickle
from torchvision import transforms

input = open('models/state/knn.pkl', 'rb')
clf = pickle.load(input)
input.close()

def transfrom_image(image):
  transform = transforms.Compose([
    # transforms.Resize((28, 28)),
    # transforms.CenterCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)),
  ])
  return transform(image).unsqueeze(0)

def predict(image):
  tensor = transfrom_image(image)
  tensor = tensor.view(-1 ,28*28)
  data = tensor.numpy()
  result = clf.predict(data)
  return float(result[0])
