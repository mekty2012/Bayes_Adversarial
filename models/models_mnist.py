# Based on https://d2l.ai/chapter_convolutional-modern/resnet.html

import copy
from math import sqrt
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

    self.bn1 = layers.Gaussian_BatchNorm2D(num_channels, var_type=var_type, init_dict=init_dict)
    self.bn2 = layers.Gaussian_BatchNorm2D(num_channels, var_type=var_type, init_dict=init_dict)
  
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

def gaussian_resnet18(var_type, init_dict, is_lrt, use_aleatoric=False):
  if is_lrt:
    b1 = nn.Sequential(
      layers.Gaussian_Conv2D_LRT(1, 64, kernel_size=7, stride=2, padding=3, var_type=var_type, init_dict=init_dict),
      layers.Gaussian_BatchNorm2D(64, var_type=var_type, init_dict=init_dict),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
  else:
    b1 = nn.Sequential(
      layers.Gaussian_Conv2D(1, 64, kernel_size=7, stride=2, padding=3, var_type=var_type, init_dict=init_dict),
      layers.Gaussian_BatchNorm2D(64, var_type=var_type, init_dict=init_dict),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
  b2 = nn.Sequential(*gaussian_resnet_block(64, 64, 2, var_type=var_type, init_dict=init_dict, is_lrt = is_lrt, first_block=True))
  b3 = nn.Sequential(*gaussian_resnet_block(64, 128, 2, var_type=var_type, init_dict=init_dict, is_lrt = is_lrt))
  b4 = nn.Sequential(*gaussian_resnet_block(128, 256, 2, var_type=var_type, init_dict=init_dict, is_lrt = is_lrt))
  b5 = nn.Sequential(*gaussian_resnet_block(256, 512, 2, var_type=var_type, init_dict=init_dict, is_lrt = is_lrt))

  if use_aleatoric:
    return nn.Sequential(b1, b2, b3, b4, b5,
                       nn.Flatten(),
                       layers.Gaussian_Linear(512, 20, var_type=var_type, init_dict=init_dict))
  else:
    return nn.Sequential(b1, b2, b3, b4, b5,
                       nn.Flatten(),
                       layers.Gaussian_Linear(512, 10, var_type=var_type, init_dict=init_dict))

class dropout_residual(nn.Module):
  def __init__(self, input_channels, num_channels, dropout_rate, dropout_type, init_dict, use_1x1conv=False, strides=1, **kwargs):
    super().__init__(**kwargs)
    self.conv1 = layers.Dropout_Conv2D(input_channels, num_channels, kernel_size=3, padding=1, stride=strides, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict)
    self.conv2 = layers.Dropout_Conv2D(num_channels, num_channels, kernel_size=3, padding=1, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict)

    if use_1x1conv:
      self.conv3 = layers.Dropout_Conv2D(input_channels, num_channels, kernel_size=1, stride=strides, dropout_rate=dropout_rate, dropout_type=dropout_type)
    else:
      self.conv3 = None
    
    self.bn1 = layers.Dropout_BatchNorm2D(num_channels, dropout_rate, dropout_type)
    self.bn2 = layers.Dropout_BatchNorm2D(num_channels, dropout_rate, dropout_type)
  
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

def dropout_resnet18(dropout_rate, dropout_type, init_dict, use_aleatoric=False):
  b1 = nn.Sequential(
      layers.Dropout_Conv2D(1, 64, kernel_size=7, stride=2, padding=3, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict),
      layers.Dropout_BatchNorm2D(64, dropout_rate, dropout_type),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
  b2 = nn.Sequential(*dropout_resnet_block(64, 64, 2, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict, first_block=True))
  b3 = nn.Sequential(*dropout_resnet_block(64, 128, 2, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict))
  b4 = nn.Sequential(*dropout_resnet_block(128, 256, 2, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict))
  b5 = nn.Sequential(*dropout_resnet_block(256, 512, 2, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict))

  if use_aleatoric:
    return nn.Sequential(b1, b2, b3, b4, b5,
                       nn.Flatten(),
                       layers.Dropout_Linear(512, 20, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict))
  else:
    return nn.Sequential(b1, b2, b3, b4, b5,
                       nn.Flatten(),
                       layers.Dropout_Linear(512, 10, dropout_rate=dropout_rate, dropout_type=dropout_type, init_dict=init_dict))

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

  def forward(self, x):
    y = nn.functional.relu(self.bn1(self.conv1(x)))
    y = self.bn2(self.conv2(y))
    if self.conv3:
      x = self.conv3(x)
    y += x
    return nn.functional.relu(y)

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

  def __init__(self, num_ensemble, use_aleatoric=False):
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
      if use_aleatoric:
        net = nn.Sequential(b1, b2, b3, b4, b5,
                          nn.Flatten(),
                          nn.Linear(512, 20))
      else:
        net = nn.Sequential(b1, b2, b3, b4, b5,
                          nn.Flatten(),
                          nn.Linear(512, 20))
      self.models.append(net)
  
  def forward(self, x):
    res = []
    for model in self.models:
      res.append(model(x))
    return torch.cat(res, dim=0)

def resnet18(use_aleatoric=False):
  b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                     nn.BatchNorm2d(64), nn.ReLU(),
                     nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
  b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
  b3 = nn.Sequential(*resnet_block(64, 128, 2))
  b4 = nn.Sequential(*resnet_block(128, 256, 2))
  b5 = nn.Sequential(*resnet_block(256, 512, 2))
  if use_aleatoric:
    net = nn.Sequential(b1, b2, b3, b4, b5,
                      nn.Flatten(),
                      nn.Linear(512, 20))
  else:
    net = nn.Sequential(b1, b2, b3, b4, b5,
                      nn.Flatten(),
                      nn.Linear(512, 10))
  return net

def get_swag_parameter(model, train_dataloader, lr, loss_fn, num_samples, frequency, rank):
  new_model = copy.deepcopy(model)
  optim = torch.optim.SGD(new_model.parameters(), lr)

  model_mean = dict()
  model_moment = dict()
  lowrank_div = []
  for name, parameter in new_model.named_parameters():
    model_mean[name] = parameter.detach().clone()
    model_moment[name] = torch.square(parameter.detach().clone())
  
  step_count = 0
  while True:
    for data in train_dataloader:
      step_count += 1
      
      images, labels = data
      optim.zero_grad()
      outputs = new_model(images)

      loss = loss_fn(outputs, labels)
      loss.backward()

      optim.step()

      if step_count % frequency == 0:
        curr_sample = step_count // frequency
        for name, parameter in new_model.named_parameters():
          model_mean[name] = (curr_sample * model_mean[name] + parameter.detach().clone()) / (curr_sample + 1)
          model_moment[name] = (curr_sample * model_moment[name] + torch.square(parameter.detach().clone())) / (curr_sample + 1)
        
        if curr_sample > num_samples - rank:
          deviation = dict()
          for name, parameter in new_model.named_parameters():
            deviation[name] = parameter.detach().clone() - model_mean[name]
          lowrank_div.append(deviation)
        
        if curr_sample == num_samples:
          model_var = dict()
          for name in model_moment:
            model_var[name] = model_moment[name] - torch.square(model_mean[name])
          return model_mean, model_var, lowrank_div

def swag_resnet18_inference(x, model_mean, model_var, lowrank_div, num_sample):
  res = []
  rank = len(lowrank_div)
  for _ in range(num_sample):
    model = resnet18()
    state = model.state_dict()
    for name in model_mean:
      eps = torch.normal(0,1, model_var[name].shape)
      new_param = model_mean[name] + torch.sqrt(model_var[name]) * eps / sqrt(2)
      for k in range(rank):
        deviation = lowrank_div[k][name]
        new_param += deviation * torch.normal(0, 1) / sqrt(2 * (rank - 1))
      state[name] = new_param
    model.load_state_dict(state)
    res.append(model(x))
  return torch.cat(res, dim=0)

class BatchEnsemble_Residual(nn.Module):
  def __init__(self, input_channels, num_channels, num_models, use_1x1conv=False, strides=1):
    super().__init__()
    self.conv1 = layers.BatchEnsemble_Conv2D(input_channels, num_channels, kernel_size=3, stride=strides,padding=1, num_models=num_models, is_first=False)
    self.conv2 = layers.BatchEnsemble_Conv2D(num_channels, num_channels, kernel_size=3, padding=1, num_models = num_models, is_first=False)

    if use_1x1conv:
      self.conv3 = layers.BatchEnsemble_Conv2D(input_channels, num_channels, kernel_size=1, stride=strides, num_models=num_models, is_first=False)
    else:
      self.conv3 = None
    
    self.bn1 = layers.BatchEnsemble_BatchNorm2D(num_channels, num_models)
    self.bn2 = layers.BatchEnsemble_BatchNorm2D(num_channels, num_models)
  
  def forward(self, x):
    y = nn.functional.relu(self.bn1(self.conv1(x)))
    y = self.bn2(self.conv2(y))
    if self.conv3:
      x = self.conv3(x)
    y += x
    return nn.functional.relu(y)

def batchensemble_resnet_block(input_channels, num_channels, num_residuals, num_models, first_block=False):
  blk = []
  for i in range(num_residuals):
    if i == 0 and not first_block:
      blk.append(BatchEnsemble_Residual(input_channels, num_channels, use_1x1conv=True, strides=2, num_models=num_models))
    else:
      blk.append(BatchEnsemble_Residual(num_channels, num_channels, num_models))
  return blk

def batchensemble_resnet18(num_models, use_aleatoric=False):
  b1 = nn.Sequential(
    layers.BatchEnsemble_Conv2D(1, 64, kernel_size=7, stride=2, padding=3, num_models=num_models, is_first=True),
    layers.BatchEnsemble_BatchNorm2D(64, num_models), 
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
  b2 = nn.Sequential(*batchensemble_resnet_block(64, 64, 2, num_models, True))
  b3 = nn.Sequential(*batchensemble_resnet_block(64, 128, 2, num_models))
  b4 = nn.Sequential(*batchensemble_resnet_block(128, 256, 2, num_models))
  b5 = nn.Sequential(*batchensemble_resnet_block(256, 512, 2, num_models))
  if use_aleatoric:
    net = nn.Sequential(b1, b2, b3, b4, b5,
                      nn.Flatten(),
                      layers.BatchEnsemble_Linear(512, 20, num_models, is_first=False))
  else:
    net = nn.Sequential(b1, b2, b3, b4, b5,
                      nn.Flatten(),
                      layers.BatchEnsemble_Linear(512, 10, num_models, is_first=False))
  return net
  