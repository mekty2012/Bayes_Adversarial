# Based on https://d2l.ai/chapter_convolutional-modern/resnet.html

import numpy as np
import torch
from torch import nn
import layers

class gaussian_residual(nn.Module):
  def __init__(self, input_channels, num_channels, var_type, init_dict, is_lrt=False, use_1x1conv=False, strides=1, **kwargs):
    super().__init__(**kwargs)
    if is_lrt:
      self.conv1 = layers.Gaussian_Conv2D_LRT(input_channels, num_channels, kernel_size=3, padding=1, stride=strides, var_type=var_type, init_dict=init_dict)
      self.conv2 = layers.Gaussian_Conv2D_LRT(num_channels, num_channels, kernel_size=3, padding=1, var_type=var_type, init_dict=init_dict)

      if use_1x1conv:
        self.conv3 = layers.Gaussian_Conv2D_LRT(input_channels, num_channels, kernel_size=1, stride=strides, var_type=var_type, init_dict=init_dict)
      else:
        self.conv3 = None
    else:
      self.conv1 = layers.Gaussian_Conv2D(input_channels, num_channels, kernel_size=3, padding=1, stride=strides, var_type=var_type, init_dict=init_dict)
      self.conv2 = layers.Gaussian_Conv2D(num_channels, num_channels, kernel_size=3, padding=1, var_type=var_type, init_dict=init_dict)

      if use_1x1conv:
        self.conv3 = layers.Gaussian_Conv2D(input_channels, num_channels, kernel_size=1, stride=strides, var_type=var_type, init_dict=init_dict)
      else:
        self.conv3 = None

    self.bn1 = nn.BatchNorm2d(num_channels)
    self.bn2 = nn.BatchNorm2d(num_channels)
  
  def forward(self, x):
    y = nn.functional.relu(self.bn1(self.conv1(x)))
    y = self.bn2(self.conv2(y))
    if self.conv3:
      x = self.conv3(x)
    y += x
    return nn.functional.relu(y)

def gaussian_resnet_block(input_channels, num_channels, num_residuals, var_type, init_dict, is_lrt, first_block=False):
  blk = []
  for i in range(num_residuals):
    if i == 0 and not first_block:
      blk.append(gaussian_residual(input_channels, num_channels, var_type=var_type, is_lrt = is_lrt, init_dict=init_dict, use_1x1conv=True, strides=2))
    else:
      blk.append(gaussian_residual(num_channels, num_channels, var_type=var_type, is_lrt = is_lrt, init_dict=init_dict))
  return blk

def gaussian_resnet18(var_type, init_dict, is_lrt):
  if is_lrt:
    b1 = nn.Sequential(
      layers.Gaussian_Conv2D_LRT(1, 64, kernel_size=7, stride=2, padding=3, var_type=var_type, init_dict=init_dict),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
  else:
    b1 = nn.Sequential(
      layers.Gaussian_Conv2D(1, 64, kernel_size=7, stride=2, padding=3, var_type=var_type, init_dict=init_dict),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
  b2 = nn.Sequential(*gaussian_resnet_block(64, 64, 2, var_type=var_type, init_dict=init_dict, is_lrt = is_lrt, first_block=True))
  b3 = nn.Sequential(*gaussian_resnet_block(64, 128, 2, var_type=var_type, init_dict=init_dict, is_lrt = is_lrt))
  b4 = nn.Sequential(*gaussian_resnet_block(128, 256, 2, var_type=var_type, init_dict=init_dict, is_lrt = is_lrt))
  b5 = nn.Sequential(*gaussian_resnet_block(256, 512, 2, var_type=var_type, init_dict=init_dict, is_lrt = is_lrt))

  return nn.Sequential(b1, b2, b3, b4, b5,
                       nn.AdaptiveAvgPool2d((1,1)),
                       nn.Flatten(),
                       layers.Gaussian_Linear(512, 20, var_type=var_type, init_dict=init_dict))

class dropout_residual(nn.Module):
  def __init__(self, input_channels, num_channels, dropout_rate, dropout_type, init_dict, use_1x1conv=False, strides=1, **kwargs):
    super().__init__(**kwargs)
    self.conv1 = layers.Dropout_Conv2D(input_channels, num_channels, kernel_size=3, padding=1, stride=strides, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict)
    self.conv2 = layers.Dropout_Conv2D(num_channels, num_channels, kernel_size=3, padding=1, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict)

    if use_1x1conv:
      self.conv3 = layers.Dropout_Conv2D(input_channels, num_channels, kernel_size=1, stride=strides, dropout_rate=dropout_rate, dropout_type=dropout_type)
    else:
      self.conv3 = None
    
    self.bn1 = nn.BatchNorm2d(num_channels)
    self.bn2 = nn.BatchNorm2d(num_channels)
  
  def forward(self, x):
    y = nn.functional.relu(self.bn1(self.conv1(x)))
    y = self.bn2(self.conv2(y))
    if self.conv3:
      x = self.conv3(x)
    y += x
    return nn.functional.relu(y)

def dropout_resnet_block(input_channels, num_channels, num_residuals, dropout_rate, dropout_type, init_dict, first_block=False):
  blk = []
  for i in range(num_residuals):
    if i == 0 and not first_block:
      blk.append(dropout_residual(input_channels, num_channels, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict, use_1x1conv=True, strides=2))
    else:
      blk.append(dropout_residual(num_channels, num_channels, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict))
  return blk

def dropout_resnet18(dropout_rate, dropout_type, init_dict):
  b1 = nn.Sequential(
      layers.Dropout_Conv2D(1, 64, kernel_size=7, stride=2, padding=3, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
  b2 = nn.Sequential(*dropout_resnet_block(64, 64, 2, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict, first_block=True))
  b3 = nn.Sequential(*dropout_resnet_block(64, 128, 2, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict))
  b4 = nn.Sequential(*dropout_resnet_block(128, 256, 2, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict))
  b5 = nn.Sequential(*dropout_resnet_block(256, 512, 2, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict))

  return nn.Sequential(b1, b2, b3, b4, b5,
                       nn.AdaptiveAvgPool2d((1,1)),
                       nn.Flatten(),
                       layers.Dropout_Linear(512, 20, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict))

class Residual(nn.Module):
  def __init__(self, input_channels, num_channels,
               use_1x1conv=False, strides=1):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channels, num_channels,
                           kernel_size=3, padding=1, stride=strides)
    self.conv2 = nn.Conv2d(num_channels, num_channels,
                           kernel_size=3, padding=1)
    if use_1x1conv:
      self.conv3 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=1, stride=strides)
    else:
      self.conv3 = None
    self.bn1 = nn.BatchNorm2d(num_channels)
    self.bn2 = nn.BatchNorm2d(num_channels)

  def forward(self, X):
    Y = nn.functional.relu(self.bn1(self.conv1(X)))
    Y = self.bn2(self.conv2(Y))
    if self.conv3:
      X = self.conv3(X)
    Y += X
    return nn.functional.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
  blk = []
  for i in range(num_residuals):
    if i == 0 and not first_block:
      blk.append(Residual(input_channels, num_channels,
                          use_1x1conv=True, strides=2))
    else:
      blk.append(Residual(num_channels, num_channels))
  return blk

class ensemble_resnet18(nn.Module):

  def __init__(self, num_ensemble):
    super().__init__()
    self.models = nn.ModuleList()
    for _ in range(num_ensemble):
      b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                         nn.BatchNorm2d(64), nn.ReLU(),
                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
      b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
      b3 = nn.Sequential(*resnet_block(64, 128, 2))
      b4 = nn.Sequential(*resnet_block(128, 256, 2))
      b5 = nn.Sequential(*resnet_block(256, 512, 2))
      net = nn.Sequential(b1, b2, b3, b4, b5,
                          nn.AdaptiveAvgPool2d((1,1)),
                          nn.Flatten(), nn.Linear(512, 20))
      self.models.append(net)
  
  def forward(self, x):
    res = []
    for model in self.models:
      res.append(model(x))
    return res