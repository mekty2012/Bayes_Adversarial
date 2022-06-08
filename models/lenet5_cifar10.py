# Based on https://deep-learning-study.tistory.com/503

import copy
from math import sqrt
import numpy as np
import torch
from torch import nn
import layers
import torch.nn.functional as F

class lenet5(nn.Module):
  def __init__(self, use_aleatoric=False):
    super(lenet5,self).__init__()
    self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3), nn.ReLU(), nn.Conv2d(32, 32, 3))
    self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.ReLU(), nn.Conv2d(64, 64, 3))
    self.conv3 = nn.Sequential(nn.Conv2d(64, 120, 3), nn.ReLU(), nn.Conv2d(120, 120, 3))
    self.dropout = nn.Dropout(0.1)
    self.fc1 = nn.Linear(120, 84)
    if use_aleatoric:
      self.fc2 = nn.Linear(84, 20)
    else:
      self.fc2 = nn.Linear(84, 10)

  def forward(self, x):
    x = torch.relu(self.conv1(x))
    x = F.avg_pool2d(x, 2, 2)
    x = self.dropout(x)
    x = torch.relu(self.conv2(x))
    x = F.avg_pool2d(x, 2, 2)
    x = self.dropout(x)
    x = torch.relu(self.conv3(x))
    x = self.dropout(x)
    x = x.view(-1, 120)
    x = torch.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x

class gaussian_lenet5(nn.Module):
  def __init__(self, var_type, init_dict, is_lrt, use_aleatoric=False):
    super().__init__()
    if is_lrt:
      self.conv1 = nn.Sequential(layers.Gaussian_Conv2D_LRT(3, 32, kernel_size=3, stride=1, var_type=var_type), nn.ReLU(), layers.Gaussian_Conv2D_LRT(32, 32, kernel_size=3, stride=1, var_type=var_type))
      self.conv2 = nn.Sequential(layers.Gaussian_Conv2D_LRT(32, 64, kernel_size=3, stride=1, var_type=var_type), nn.ReLU(), layers.Gaussian_Conv2D_LRT(64, 64, kernel_size=3, stride=1, var_type=var_type))
      self.conv3 = nn.Sequential(layers.Gaussian_Conv2D_LRT(64, 120, kernel_size=3, stride=1, var_type=var_type), nn.ReLU(), layers.Gaussian_Conv2D_LRT(120, 120, kernel_size=3, stride=1, var_type=var_type))
    else:
      self.conv1 = nn.Sequential(layers.Gaussian_Conv2D(3, 32, kernel_size=3, stride=1, var_type=var_type), nn.ReLU(), layers.Gaussian_Conv2D_LRT(32, 32, kernel_size=3, stride=1, var_type=var_type))
      self.conv2 = nn.Sequential(layers.Gaussian_Conv2D(32, 64, kernel_size=3, stride=1, var_type=var_type), nn.ReLU(), layers.Gaussian_Conv2D_LRT(64, 64, kernel_size=3, stride=1, var_type=var_type))
      self.conv3 = nn.Sequential(layers.Gaussian_Conv2D(64, 120, kernel_size=3, stride=1, var_type=var_type), nn.ReLU(), layers.Gaussian_Conv2D_LRT(120, 120, kernel_size=3, stride=1, var_type=var_type))
    self.fc1 = layers.Gaussian_Linear(120, 84, var_type)
    if use_aleatoric:
      self.fc2 = layers.Gaussian_Linear(84, 20, var_type)
    else:
      self.fc2 = layers.Gaussian_Linear(84, 10, var_type)
  
  def forward(self, x):
    x = torch.relu(self.conv1(x))
    x = F.avg_pool2d(x, 2, 2)
    x = torch.relu(self.conv2(x))
    x = F.avg_pool2d(x, 2, 2)
    x = torch.relu(self.conv3(x))
    x = x.view(-1, 120)
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x

class dropout_lenet5(nn.Module):
  def __init__(self, dropout_rate, dropout_type, init_dict, use_aleatoric=False):
    super().__init__()
    self.conv1 = nn.Sequential(layers.Dropout_Conv2D(3, 32, 3, dropout_rate, dropout_type, stride=1), nn.ReLU(), layers.Dropout_Conv2D(32, 32, 3, dropout_rate, dropout_type, stride=1))
    self.conv2 = nn.Sequential(layers.Dropout_Conv2D(32, 64, 3, dropout_rate, dropout_type, stride=1), nn.ReLU(), layers.Dropout_Conv2D(64, 64, 3, dropout_rate, dropout_type, stride=1))
    self.conv3 = nn.Sequential(layers.Dropout_Conv2D(64, 120, 3, dropout_rate, dropout_type, stride=1), nn.ReLU(), layers.Dropout_Conv2D(120, 120, 3, dropout_rate, dropout_type, stride=1))
    self.fc1 = layers.Dropout_Linear(120, 84, dropout_rate, dropout_type)
    if use_aleatoric:
      self.fc2 = layers.Dropout_Linear(84, 20, dropout_rate, dropout_type)
    else:
      self.fc2 = layers.Dropout_Linear(84, 10, dropout_rate, dropout_type)
  
  def forward(self, x):
    x = torch.relu(self.conv1(x))
    x = F.avg_pool2d(x, 2, 2)
    x = torch.relu(self.conv2(x))
    x = F.avg_pool2d(x, 2, 2)
    x = torch.relu(self.conv3(x))
    x = x.view(-1, 120)
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x
  
class ensemble_lenet5(nn.Module):
  def __init__(self, num_ensemble, use_aleatoric=False):
    super().__init__()
    self.models = nn.ModuleList()
    for _ in range(num_ensemble):
      self.models.append(lenet5(use_aleatoric))
  
  def forward(self, x):
    res = []
    for model in self.models:
      res.append(model(x))
    return torch.cat(res, dim=0)

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

def swag_lenet5_inference(x, model_mean, model_var, lowrank_div, num_sample):
  res = []
  rank = len(lowrank_div)
  for _ in range(num_sample):
    model = lenet5()
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

class batchensemble_lenet5(nn.Module):
  def __init__(self, num_ensemble, use_aleatoric=False):
    super().__init__()
    self.conv1 = nn.Sequential(layers.BatchEnsemble_Conv2D(3, 32, 3, num_models=num_ensemble, stride=1, is_first=True), nn.ReLU(), layers.BatchEnsemble_Conv2D(32, 32, 3, num_models=num_ensemble, stride=1, is_first=False))
    self.conv2 = nn.Sequential(layers.BatchEnsemble_Conv2D(32, 64, 3, num_models=num_ensemble, stride=1, is_first=False), nn.ReLU(), layers.BatchEnsemble_Conv2D(64, 64, 3, num_models=num_ensemble, stride=1, is_first=False))
    self.conv3 = nn.Sequential(layers.BatchEnsemble_Conv2D(64, 120, 3, num_models=num_ensemble, stride=1, is_first=False), nn.ReLU(), layers.BatchEnsemble_Conv2D(120, 120, 3, num_models=num_ensemble, stride=1, is_first=False))
    self.dropout = nn.Dropout(0.1)
    self.fc1 = layers.BatchEnsemble_Linear(120, 84, num_models=num_ensemble, is_first=False)
    if use_aleatoric:
      self.fc2 = layers.BatchEnsemble_Linear(84, 20, num_models=num_ensemble, is_first=False)
    else:
      self.fc2 = layers.BatchEnsemble_Linear(84, 10, num_models=num_ensemble, is_first=False)
  
  def forward(self, x):
    x = torch.relu(self.conv1(x))
    x = F.avg_pool2d(x, 2, 2)
    x = self.dropout(x)
    x = torch.relu(self.conv2(x))
    x = F.avg_pool2d(x, 2, 2)
    x = self.dropout(x)
    x = torch.relu(self.conv3(x))
    x = self.dropout(x)
    x = x.view(-1, 120)
    x = torch.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x