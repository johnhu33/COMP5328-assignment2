#!/usr/bin/env python3
"""Build transition matrix estimators"""
import csv
import os
import random
import sys
from typing import Callable, List, Tuple
import lightgbm as lgb
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torchvision

class Bottleneck(nn.Module):
  expansion = 4
  
  def __init__(self, in_dim, out_dim, identity_shortcut = None, stride=1):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=1)
    self.bn1 = nn.BatchNorm2d(out_dim)
    self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3,stride=stride, padding=1)
    self.bn2 = nn.BatchNorm2d(out_dim)
    self.conv3 = nn.Conv2d(out_dim, self.expansion*out_dim, kernel_size=1)
    self.bn3 = nn.BatchNorm2d(self.expansion*out_dim)
    self.relu = nn.ReLU()
    self.identity_shortcut = identity_shortcut
  
  def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv3(out)
    out = self.bn3(out)

    if self.identity_shortcut is not None:
      identity = self.identity_shortcut(identity)

    out += identity
    out = self.relu(out)
    return out

class ResNet(nn.Module):
  def __init__(self, block, num_blocks, num_classes,image_channels):
    super(ResNet, self).__init__()
    self.in_planes = 64
    self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7,stride=1, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding =1)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512*4,num_classes)

  def _make_layer(self, block, out_dim, num_blocks, stride):
    identity_shortcut = None
    layers = []
    if stride != 1 or self.in_dim != out_dim*4:
      identity_shortcut = nn.Sequential(nn.Conv2d(self.in_dim,out_dim*4,kernel_size=1,stride=stride), nn.BatchNorm2d(out_dim*4))
    layers.append(block(self.in_dim, out_dim, identity_shortcut, stride))
    self.in_dim = out_dim*4

    for i in range (num_blocks -1):
      layers.append(block(self.in_dim,out_dim))
      
    return nn.Sequential(*layers)


  def forward(self, x):
      out = torch.relu(self.bn1(self.conv1(x)))
      out = self.layer1(out)
      out = self.layer2(out)
      out = self.layer3(out)
      out = self.layer4(out)
      out = self.avgpool(out)
      out = out.view(out.size(0), -1)
      out = self.fc(out)
      return out


def Res50 (num_layers, num_classes = 3, image_channels =3):
  return ResNet(Bottleneck, [3,4,6,3], image_channels, num_classes)
def Res101 (num_layers, num_classes = 3, image_channels =3):
  return ResNet(Bottleneck, [3,4,23,3], image_channels, num_classes)
def Res152 (num_layers, num_classes = 3, image_channels =3):
  return ResNet(Bottleneck, [3,8,36,3], image_channels, num_classes)

class Backward:
    def __init__(self, model):
        self._model = model

    def train(self, X: np.ndarray, y: np.ndarray, _: np.ndarray) -> None:
        self._model.fit(X, y)

    def __call__(self,
                 X: np.ndarray,
                 T: np.ndarray,
                 denoise: bool = False) -> np.ndarray:
        ret = self._model.predict_proba(X)
        if denoise:
            ret = softmax(np.linalg.inv(T) @ ret.T, axis=0).T
        return ret


Model = Callable[[int, int], nn.Module]


class Forward:
    def __init__(self, build: Model):
        self._build = build

    def train(self, X: np.ndarray, y: np.ndarray, T: np.ndarray) -> None:
        T = torch.from_numpy(T.astype(np.float32))
        sm = nn.Softmax(dim=1)
        self._model = train(self._build, X, y, lambda x: sm(T @ sm(x).T).T)

    def __call__(self,
                 X: np.ndarray,
                 T: np.ndarray,
                 denoise: bool = False) -> np.ndarray:
        with torch.no_grad():
            ret = softmax(self._model(torch.from_numpy(X.astype(
                np.float32))).numpy(),
                          axis=1)
        if not denoise:
            ret = softmax(T @ ret.T, axis=0).T
        return ret


def train(build: Model, X: np.ndarray, y: np.ndarray,
          transform: Callable[[torch.Tensor], torch.Tensor]) -> nn.Module:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build(X.shape[1], max(y) + 1)
    if torch.cuda.device_count() > 1:
        model = nn.DistributedDataParallel(model)
    model.to(device)
    X = torch.from_numpy(X.astype(np.float32)).to(device)
    y = torch.from_numpy(y.astype(np.int64)).to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-1,
                                weight_decay=1e-5,
                                momentum=0.9)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        X, y),
                                               batch_size=256,
                                               shuffle=True)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = transform(model(X))
            criterion(pred, y).backward()
            optimizer.step()
    model.eval()
    return model


class NeuralNet:
    def __init__(self, build: Model):
        self._build = build

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = train(self._build, X, y, lambda x: x)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return softmax(self._model(torch.from_numpy(X.astype(
                np.float32))).numpy(),
                           axis=1)


def evaluate(dataset: str, T: List[List[float]], model) -> Tuple[float, float]:
    with np.load(f'data/{dataset}.npz') as data:
        Xtr = data['Xtr'].reshape((len(data['Xtr']), -1))
        Xts = data['Xts'].reshape((len(data['Xts']), -1))
        Xtr, Xtr_val, Str, Str_val = train_test_split(Xtr,
                                                      data['Str'],
                                                      test_size=0.2)
        Yts = data['Yts']
    T = np.array(T)
    model.train(Xtr, Str, T)
    acc_val = top1_accuracy(model(Xtr_val, T), Str_val)
    acc = top1_accuracy(model(Xts, T, True), Yts)
    return acc_val, acc


def linear(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Linear(in_dim, out_dim)


def three_layer(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                         nn.Linear(out_dim, out_dim), nn.ReLU(),
                         nn.Linear(out_dim, out_dim))


def top1_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return sum(pred.argmax(axis=1) == y) / len(y)


def reset_seed(seed: int = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # If multi-GPUs are used.
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main() -> None:
    """Run all training and evaluation"""
    w = csv.DictWriter(
        sys.stdout,
        ['dataset', 'model', 'acc_val', 'acc_val_std', 'acc', 'acc_std'])
    w.writeheader()
    for dataset, T in DATA.items():
        for name, model in MODEL.items():
            reset_seed()
            acc_val, acc = [], []
            for i in range(10):
                v, a = evaluate(dataset, T, model)
                acc_val.append(v)
                acc.append(a)
            w.writerow({
                'dataset': dataset,
                'model': name,
                'acc_val': np.mean(acc_val),
                'acc_val_std': np.std(acc_val),
                'acc': np.mean(acc),
                'acc_std': np.std(acc)
            })


DATA = {
    'FashionMNIST0.5': [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]],
    'FashionMNIST0.6': [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]],
}
MODEL = {
    'Forward_Res50':Forward(NeuralNet(Res50))
    'Forward_Res101':Forward(NeuralNet(Res101))
    'Forward_Res152':Forward(NeuralNet(Res152))
    'forward_linear': Forward(linear),
    'backward_linear': Backward(NeuralNet(linear)),
    'forward_three_layer': Forward(three_layer),
    'backward_three_layer': Backward(NeuralNet(three_layer)),
    'LGB': Backward(lgb.LGBMClassifier()),
    'logistic': Backward(LogisticRegression()),
}

if __name__ == '__main__':
    main()
