import numpy as np
import torch
import torch.nn as nn
from math import sqrt

class Gaussian_Linear(nn.Module):
  """
  Implements the Linear layer for BNN.
  Each W, B are Gaussian random variable, except that, it is equivalent to nn.Linear.
  This uses Local Reparameterization Trick, which accelerates the computation.

  var_type : Choose the mode for the variance. 
             "sq"  : variance = sigma^2
             "exp" : variance = exp(sigma / 2)  (Same to the VAE)
  init_dict : Initialization mean/variances for the parameters. By default, He init.
  """
  def __init__(self, 
              in_features, # Input dimension 
              out_features, # Output dimension
              var_type, # Variance model. Use square(sigma) or exp(sigma).
              bias=True, # Use bias or not.
              device=None, # Device of layer.
              dtype=None, # Datatype of layer.
              init_dict=dict() # The dictionary of parameters.
              ):
    super(Gaussian_Linear, self).__init__()
    k = sqrt(1 / in_features)
    self.init_dict = {"w_mu_mean" : 0,  "w_mu_std" : k, 
                      "w_sig_mean" : 0, "w_sig_std" : k, 
                      "b_mu_mean" : 0,  "b_mu_std" : k, 
                      "b_sig_mean" : 0, "b_sig_std" : k}
    for k, v in init_dict.items():
      self.init_dict[k] = v
    
    self.use_bias = bias
    self.weight_mu = nn.Parameter(data=torch.normal(self.init_dict["w_mu_mean"], self.init_dict["w_mu_std"], [in_features, out_features], dtype=dtype, device=device))
    self.weight_sigma = nn.Parameter(data=torch.normal(self.init_dict["w_sig_mean"], self.init_dict["w_sig_std"], [in_features, out_features], dtype=dtype, device=device))
    if self.use_bias:
      self.bias_mu = nn.Parameter(data=torch.normal(self.init_dict["b_mu_mean"], self.init_dict["b_mu_std"], [out_features], dtype=dtype, device=device))
      self.bias_sigma = nn.Parameter(data=torch.normal(self.init_dict["b_sig_mean"], self.init_dict["b_sig_std"], [out_features], dtype=dtype, device=device))
    if var_type != "exp" and var_type != "sq":
      raise ValueError("The variance mode should be exp or sq.")
    self.var_type = var_type
    
  def forward(self, x):
    if self.use_bias:
      new_mean = torch.matmul(x, self.weight_mu) + self.bias_mu
      if self.var_type == "exp":
        new_var = torch.matmul(torch.square(x), torch.exp(self.weight_sigma)) + torch.exp(self.bias_sigma)
      elif self.var_type == "sq":
        new_var = torch.matmul(torch.square(x), torch.square(self.weight_sigma)) + torch.square(self.bias_sigma)
    else:
      new_mean = torch.matmul(x, self.weight_mu)
      if self.var_type == "exp":
        new_var = torch.matmul(torch.square(x), torch.exp(self.weight_sigma))
      elif self.var_type == "sq":
        new_var = torch.matmul(torch.square(x), torch.square(self.weight_sigma))
    
    eps = torch.normal(0, 1, new_mean.shape)
    return new_mean + eps * torch.sqrt(new_var)

class Gaussian_Conv2D_LRT(nn.Module):
  """
  Implements the Conv2d layer for BNN.
  Each W, B are Gaussian random variable, except that, it is equivalent to nn.Conv2D.
  This uses Local Reparameterization Trick, which accelerates the computation, but not accurate for Conv2D.

  var_type : Choose the mode for the variance. 
             "sq"  : variance = sigma^2
             "exp" : variance = exp(sigma / 2)  (Same to the VAE)
  init_dict : Initialization mean/variances for the parameters. By default, He init.
  """
  def __init__(self,
               in_channels, # Number of input channel
               out_channels, # Number of output channel
               kernel_size, 
               var_type, 
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               bias=True,
               device=None,
               dtype=None,
               init_dict=dict() # The dictionary of parameters.
               ):
    super(Gaussian_Conv2D_LRT, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    if isinstance(kernel_size, tuple) and len(kernel_size) >= 2:
      self.kernel_size = kernel_size
    elif isinstance(kernel_size, tuple):
      self.kernel_size = (kernel_size[0], kernel_size[0])
    else:
      self.kernel_size = (kernel_size, kernel_size)
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.var_type = var_type
    self.use_bias = bias

    k = sqrt(groups / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
    self.init_dict = {"w_mu_mean" : 0,  "w_mu_std" : k, 
                      "w_sig_mean" : 0, "w_sig_std" : k, 
                      "b_mu_mean" : 0,  "b_mu_std" : k, 
                      "b_sig_mean" : 0, "b_sig_std" : k}
    for k, v in init_dict.items():
      self.init_dict[k] = v
    

    if self.in_channels % self.groups != 0:
      raise ValueError("in_channels must be divisible by groups")
    shape = [self.out_channels, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]]
    self.weight_mu = nn.Parameter(data=torch.normal(self.init_dict["w_mu_mean"], self.init_dict["w_mu_std"], shape, dtype=dtype, device=device))
    self.weight_sigma = nn.Parameter(data=torch.normal(self.init_dict["w_sig_mean"], self.init_dict["w_sig_std"], shape, dtype=dtype, device=device))
    if self.use_bias:
      self.bias_mu = nn.Parameter(data=torch.normal(self.init_dict["b_mu_mean"], self.init_dict["b_mu_std"], [self.out_channels], dtype=dtype, device=device))
      self.bias_sigma = nn.Parameter(data=torch.normal(self.init_dict["b_sig_mean"], self.init_dict["b_sig_std"], [self.out_channels], dtype=dtype, device=device))
    

  def forward(self, x):
    if self.use_bias:
      new_mean = nn.functional.conv2d(x, self.weight_mu, self.bias_mu, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
      if self.var_type == "exp":
        new_var = nn.functional.conv2d(torch.square(x), torch.exp(self.weight_sigma), torch.exp(self.bias_mu), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
      else:
        new_var = nn.functional.conv2d(torch.square(x), torch.square(self.weight_sigma), torch.square(self.bias_sigma), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
    else:
      new_mean = nn.funcitonal.conv2d(x, self.weight_mu, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
      if self.var_type == "exp":
        new_var = nn.functional.conv2d(torch.square(x), torch.exp(self.weight_sigma), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
      else:
        new_var = nn.functional.conv2d(torch.square(x), torch.square(self.weight_sigma), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
    eps = torch.normal(0, 1, new_mean.shape)
    return new_mean + eps * torch.sqrt(new_var)

class Gaussian_Conv2D(nn.Module):
  """
  Implements the Conv2d layer for BNN.
  Each W, B are Gaussian random variable, except that, it is equivalent to nn.Conv2D.
  This do not use the Local Reparameterization Trick. It is accurate, but not fast. (It iterates the batch.)

  var_type : Choose the mode for the variance. 
             "sq"  : variance = sigma^2
             "exp" : variance = exp(sigma / 2)  (Same to the VAE)
  init_dict : Initialization mean/variances for the parameters. By default, He init.
  """
  def __init__(self,
               in_channels, # Number of input channel
               out_channels, # Number of output channel
               kernel_size, 
               var_type, 
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               bias=True,
               device=None,
               dtype=None,
               init_dict=dict() # The dictionary of parameters.
               ):
    super(Gaussian_Conv2D, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    if isinstance(kernel_size, tuple) and len(kernel_size) >= 2:
      self.kernel_size = kernel_size
    elif isinstance(kernel_size, tuple):
      self.kernel_size = (kernel_size[0], kernel_size[0])
    else:
      self.kernel_size = (kernel_size, kernel_size)
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.var_type = var_type
    self.use_bias = bias
    
    k = sqrt(groups / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
    self.init_dict = {"w_mu_mean" : 0,  "w_mu_std" : k, 
                      "w_sig_mean" : 0, "w_sig_std" : k, 
                      "b_mu_mean" : 0,  "b_mu_std" : k, 
                      "b_sig_mean" : 0, "b_sig_std" : k}
    for k, v in init_dict.items():
      self.init_dict[k] = v
    
    if self.in_channels % self.groups != 0:
      raise ValueError("in_channels must be divisible by groups")
    shape = [self.out_channels, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]]
    self.weight_mu = nn.Parameter(data=torch.normal(self.init_dict["w_mu_mean"], self.init_dict["w_mu_std"], shape, dtype=dtype, device=device))
    self.weight_sigma = nn.Parameter(data=torch.normal(self.init_dict["w_sig_mean"], self.init_dict["w_sig_std"], shape, dtype=dtype, device=device))
    if self.use_bias:
      self.bias_mu = nn.Parameter(data=torch.normal(self.init_dict["b_mu_mean"], self.init_dict["b_mu_std"], [self.out_channels], dtype=dtype, device=device))
      self.bias_sigma = nn.Parameter(data=torch.normal(self.init_dict["b_sig_mean"], self.init_dict["b_sig_std"], [self.out_channels], dtype=dtype, device=device))
    
  def forward(self, x):
    batch_size = x.shape[0]
    res = []
    if self.var_type == "exp":
      weight_std = torch.exp(self.weight_sigma / 2)
      if self.use_bias:
        bias_std = torch.exp(self.bias_sigma / 2)
    else:
      weight_std = self.weight_sigma
      if self.use_bias:
        bias_std = self.bias_sigma
    for i in range(batch_size):
      xi = torch.unsqueeze(x[i, :, :, :], dim=0)
      # [1, channel, height, width]
      if self.use_bias:
        new_weight = torch.normal(self.weight_mu, torch.abs(weight_std))
        new_bias = torch.normal(self.bias_mu, torch.abs(bias_std))
        yi = nn.functional.conv2d(xi, new_weight, new_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        res.append(yi)
      else:
        new_weight = torch.normal(self.weight_mu, torch.abs(weight_std))
        yi = nn.functional.conv2d(xi, new_weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        res.append(yi)
    return torch.concat(res, dim=0)

class Gaussian_BatchNorm2D(nn.Module):
  """
  Implements the BatchNorm2D for BNN.
  The gamma and beta parameters are Gaussians now.
  """
  def __init__(self,
               num_features,
               var_type, 
               eps=1e-05,
               momentum=0.1,
               track_running_stats=True,
               device=None,
               dtype=None,
               init_dict=dict()):
    super(Gaussian_BatchNorm2D, self).__init__()
    self.batch_norm = nn.BatchNorm2d(num_features, eps, momentum, affine=False, track_running_stats=track_running_stats, device=device, dtype=dtype)
    self.init_dict = {"gamma_mu_mean" : 1, "gamma_mu_std" : 0.1,
                      "gamma_sigma_mean" : 0, "gamma_sigma_std" : 1,
                      "beta_mu_mean" : 0, "beta_mu_std" : 0.1,
                      "beta_sigma_mean" : 0, "beta_sigma_std" : 1}
    for k, v in init_dict.items():
      self.init_dict[k] = v
    self.gamma_mu = nn.Parameter(torch.normal(self.init_dict["gamma_mu_mean"], self.init_dict["gamma_mu_std"], [1, num_features, 1, 1], device=device, dtype=dtype))
    self.gamma_sigma = nn.Parameter(torch.normal(self.init_dict["gamma_sigma_mean"], self.init_dict["gamma_sigma_std"], [1, num_features, 1, 1], device=device, dtype=dtype))
    self.beta_mu = nn.Parameter(torch.normal(self.init_dict["beta_mu_mean"], self.init_dict["beta_mu_std"], [1, num_features, 1, 1], device=device, dtype=dtype))
    self.beta_sigma = nn.Parameter(torch.normal(self.init_dict["beta_sigma_mean"], self.init_dict["beta_sigma_std"], [1, num_features, 1, 1], device=device, dtype=dtype))
    self.var_type = var_type
  
  def forward(self, x):
    batch_size = x.shape[0]
    num_channel = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    
    gamma_mu = self.gamma_mu.repeat([batch_size, 1, height, width])
    gamma_sigma = self.gamma_sigma.repeat([batch_size, 1, height, width])
    beta_mu = self.beta_mu.repeat([batch_size, 1, height, width])
    beta_sigma = self.beta_sigma.repeat([batch_size, 1, height, width])
    
    normed = self.batch_norm(x)
    new_mean = normed * gamma_mu + beta_mu
    
    if self.var_type == "sq":
      new_var = torch.square(normed) * torch.square(gamma_sigma) + torch.square(beta_sigma)
    else:
      new_var = torch.square(normed) * torch.exp(gamma_sigma) + torch.exp(beta_sigma)
    eps = torch.normal(0, 1, new_mean.shape)
    return new_mean + eps * torch.sqrt(new_var)

class Dropout_Linear(nn.Module):
  """
  Implements the Linear layer for MC-Dropout.
  Each W, B are dropouted, except that, it is equivalent to nn.Linear.
  
  dropout_rate : The rate of dropout. When 0, no dropout. 
  dropout_type : Mode of dropout.
                 "w" : Dropout the weight. It makes computation iteration over batch (slow).
                 "f" : Dropout the output feature. Faster, and is equivalent to dropping row-wise.
  init_dict : Initialization mean/variances for the parameters. By default, He init.
  """
  def __init__(self,
               in_features,
               out_features,
               dropout_rate,
               dropout_type,
               bias=True,
               device=None,
               dtype=None,
               init_dict=dict()):
    
    super(Dropout_Linear, self).__init__()
    k = sqrt(1 / in_features)
    self.init_dict = {"w_mean" : 0, "w_std" : k,
                      "b_mean" : 0, "b_std" : k}
    for k, v in init_dict.items():
      self.init_dict[k] = v

    self.use_bias = bias

    self.weight = nn.Parameter(data=torch.normal(self.init_dict["w_mean"], self.init_dict["w_std"], [in_features, out_features], dtype=dtype, device=device))
    if self.use_bias:
      self.bias = nn.Parameter(data=torch.normal(self.init_dict["b_mean"], self.init_dict["b_std"], [out_features], dtype=dtype, device=device))
    self.dropout_rate = dropout_rate
    if dropout_type not in ["w", "f", "c"]:
      raise ValueError("dropout_type should be either w(weight), f(feature), c(channel).")
    self.dropout_type=dropout_type
  
  def dropout(x, p):
    return x * (torch.rand(x.shape) > p)
  
  def forward(self, x):
    if self.dropout_type == "w":
      res = []
      batch_size = x.shape[0]
      for i in range(batch_size):
        xi = torch.unsqueeze(x[i, :], 0)
        if self.use_bias:
          new_w = Dropout_Linear.dropout(self.weight, self.dropout_rate)
          new_b = Dropout_Linear.dropout(self.bias, self.dropout_rate)
          yi = torch.matmul(xi, new_w) + new_b
          res.append(yi)
        else:
          new_w = Dropout_Linear.dropout(self.weight, self.dropout_rate)
          yi = torch.matmul(xi, new_w)
          res.append(yi)
      return torch.concat(res, dim=0)
    else:
      if self.use_bias:
        output = torch.matmul(x, self.weight) + self.bias
        return Dropout_Linear.dropout(output, self.dropout_rate)
      else:
        output = torch.matmul(x, self.weight)
        return Dropout_Linear.dropout(output, self.dropout_rate)

class Dropout_Conv2D(nn.Module):
  """
  Implements the Conv2D layer for MC-Dropout.
  Each W, B are dropouted, except that, it is equivalent to nn.Conv2D.
  
  dropout_rate : The rate of dropout. When 0, no dropout. 
  dropout_type : Mode of dropout.
                 "w" : Dropout the weight. It makes computation iteration over batch (slow).
                 "f" : Dropout the output feature. Faster, and is equivalent to dropping row-wise.
                 "c" : Dropout the output channel. Faster, and is equivalent to droppint all channel weight.
  init_dict : Initialization mean/variances for the parameters. By default, He init.
  """
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               dropout_rate,
               dropout_type,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               bias=True,
               padding_mode='zeros',
               device=None,
               dtype=None,
               init_dict=dict() # The dictionary of parameters.
               ):
    super(Dropout_Conv2D, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    if isinstance(kernel_size, tuple) and len(kernel_size) >= 2:
      self.kernel_size = kernel_size
    elif isinstance(kernel_size, tuple):
      self.kernel_size = (kernel_size[0], kernel_size[0])
    else:
      self.kernel_size = (kernel_size, kernel_size)
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.use_bias = bias
    self.padding_mode = padding_mode
    self.dropout_rate = dropout_rate
    self.dropout_type = dropout_type
    if self.dropout_type not in ["w", "f", "c"]:
      raise ValueError("dropout_type should be either w(weight), f(feature), c(channel).")
    k = sqrt(groups / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
    self.init_dict = {"w_mean" : 0, "w_std" : k,
                      "b_mean" : 0, "b_std" : k}
    for k, v in init_dict.items():
      self.init_dict[k] = v
    
    if self.in_channels % self.groups != 0:
      raise ValueError("in_channels must be divisible by groups")
    shape = [self.out_channels, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]]
    
    self.weight = nn.Parameter(data=torch.normal(self.init_dict["w_mean"], self.init_dict["w_std"], shape, dtype=dtype, device=device))
    
    if self.use_bias:
      self.bias = nn.Parameter(data=torch.normal(self.init_dict["b_mean"], self.init_dict["b_std"], [out_channels], dtype=dtype, device=device))

  def dropout(x, p):
    return x * (torch.rand(x.shape) > p)

  def forward(self, x):
    if self.dropout_type == "w":
      res = []
      batch_size = x.shape[0]
      for i in range(batch_size):
        xi = torch.unsqueeze(x[i, :, :, :], 0)
        if self.use_bias:
          new_w = Dropout_Conv2D.dropout(self.weight, self.dropout_rate)
          new_b = Dropout_Conv2D.dropout(self.bias, self.dropout_rate)
          yi = nn.functional.conv2d(xi, new_w, new_b, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
          res.append(yi)
        else:
          new_w = Dropout_Conv2D.dropout(self.weight, self.dropout_rate)
          yi = nn.functional.conv2d(xi, new_w, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
          res.append(yi)
      return torch.concat(res, dim=0)
    elif self.dropout_type == "f":
      if self.use_bias:
        output = nn.functional.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
      else:
        output = nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
      return Dropout_Conv2D.dropout(output, self.dropout_rate)
    else:
      if self.use_bias:
        output = nn.functional.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
      else:
        output = nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
      return nn.functional.dropout2d(output, self.dropout_rate)

class Dropout_BatchNorm2D(nn.Module):

  def __init__(self,
               num_features,
               dropout_rate,
               dropout_type,
               eps=1e-05,
               momentum=0.1,
               track_running_stats=True,
               device=None,
               dtype=None,
               init_dict=dict()):
    super(Dropout_BatchNorm2D, self).__init__()
    self.batch_norm = nn.BatchNorm2d(num_features, eps, momentum, affine=False, track_running_stats=track_running_stats, device=device, dtype=dtype)
    self.init_dict = {"gamma_mean" : 1, "gamma_std" : 0,
                      "beta_mean" : 0, "beta_std" : 0}
    for k, v in init_dict.items():
      self.init_dict[k] = v
    
    self.gamma = nn.Parameter(torch.normal(self.init_dict["gamma_mean"], self.init_dict["gamma_std"], [1, num_features, 1, 1], device=device, dtype=dtype))
    self.beta = nn.Parameter(torch.normal(self.init_dict["beta_mean"], self.init_dict["beta_std"], [1, num_features, 1, 1], device=device, dtype=dtype))

    self.dropout_rate = dropout_rate
    self.dropout_type = dropout_type
  
  def dropout(x, p):
    return x * (torch.rand(x.shape) > p)

  def forward(self, x):
    batch_size = x.shape[0]
    num_channel = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]

    normed = self.batch_norm(x)
    if self.dropout_type == "w":
      res = []
      gamma = self.gamma.repeat([1, 1, height, width])
      beta = self.beta.repeat([1, 1, height, width])
      for i in range(batch_size):
        xi = torch.unsqueeze(normed[i, :, :, :], 0)
        yi = xi * Dropout_BatchNorm2D.dropout(gamma, self.dropout_rate) + Dropout_BatchNorm2D.dropout(beta, self.dropout_rate)
        res.append(yi)
      return torch.concat(res, 0)
    else:
      gamma = self.gamma.repeat([batch_size, 1, height, width])
      beta = self.beta.repeat([batch_size, 1, height, width])
      y = normed * gamma + beta
      return Dropout_BatchNorm2D.dropout(y, self.dropout_rate)

class BatchEnsemble_Linear(nn.Module):

  def __init__(self, in_features, out_features, is_first, num_models, bias=True, device=None, dtype=None):
    super().__init__()
    self.fc = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
    self.alpha = nn.Parameter(torch.normal(0, 1, [num_models, in_features], device=device, dtype=dtype))
    self.gamma = nn.Parameter(torch.normal(0, 1, [num_models, out_features], device=device, dtype=dtype))
    self.use_bias = bias
    if self.use_bias:
      self.bias = nn.Parameter(torch.normal(0, 1, [num_models, out_features], device=device, dtype=dtype))
    self.is_first = is_first
    self.num_models = num_models
    self.in_features = in_features
    self.out_features = out_features

  def forward(self, x):
    # The bahavior differs whether it is training or testing.
    if self.is_first:
      x = torch.cat([x for _ in range(self.num_models)], dim=0)
      
    batch_size = x.shape[0] // self.num_models
    alpha = torch.cat([self.alpha for _ in range(batch_size)], dim=1).view(-1, self.in_features)
    gamma = torch.cat([self.gamma for _ in range(batch_size)], dim=1).view(-1, self.out_features)
    if self.use_bias:
      bias = torch.cat([self.bias for _ in range(batch_size)], dim=1).view(-1, self.out_features)
      return self.fc(x * alpha) * gamma + bias
    else:
      return self.fc(x * alpha) * gamma

class BatchEnsemble_Conv2D(nn.Module):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               dilation=1, 
               padding=0,
               groups=1,
               is_first=False,
               num_models=100,
               bias=True,
               device=None,
               dtype=None):
    super().__init__()
    self.is_first = is_first
    self.use_bias = bias
    self.num_models = num_models
    self.in_channels = in_channels
    self.out_channels = out_channels
    
    self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False, padding_mode="zeros", device=device, dtype=dtype)
    self.alpha = nn.Parameter(torch.normal(0, 1, [num_models, in_channels], device=device, dtype=dtype))
    self.gamma = nn.Parameter(torch.normal(0, 1, [num_models, out_channels], device=device, dtype=dtype))
    if self.use_bias:
      self.bias = nn.Parameter(torch.normal(0, 1, [num_models, out_channels], device=device, dtype=dtype))
  
  def forward(self, x):
    if self.is_first:
      x = torch.cat([x for _ in range(self.num_models)], dim=0)

    height = x.shape[2]
    width = x.shape[3]  
    
    batch_size = x.shape[0] // self.num_models
    alpha = torch.cat([self.alpha for _ in range(batch_size)], dim=1).view([-1, self.in_channels, 1, 1]).repeat(1, 1, height, width)

    x = self.conv2d(x * alpha)

    new_height = x.shape[2]
    new_width = x.shape[3]

    gamma = torch.cat([self.gamma for _ in range(batch_size)], dim=1).view([-1, self.out_channels, 1, 1]).repeat(1, 1, new_height, new_width)
    if self.use_bias:
      bias = torch.cat([self.bias for _ in range(batch_size)], dim=1).view([-1, self.out_channels, 1, 1]).repeat(1, 1, new_height, new_width)
      return x * gamma + bias
    else:
      return x * gamma

class BatchEnsemble_BatchNorm2D(nn.Module):

  def __init__(self,
               num_features,
               num_models,
               eps=1e-05,
               momentum=0.1,
               track_running_stats=True,
               device=None,
               dtype=None):
    super().__init__()
    self.batch_norm = nn.BatchNorm2d(num_features, eps, momentum, affine=False, track_running_stats=track_running_stats, device=device, dtype=dtype)

    self.num_models = num_models
    self.num_features = num_features
    
    self.gamma = nn.Parameter(torch.ones([self.num_models, self.num_features], device=device, dtype=dtype))
    self.beta = nn.Parameter(torch.ones([self.num_models, self.num_features], device=device, dtype=dtype))
    
  def forward(self, x):
    batch_size = x.shape[0] // self.num_models
    normed = self.batch_norm(x)
    
    height = x.shape[2]
    width = x.shape[3]  
    
    gamma = torch.cat([self.gamma for _ in range(batch_size)], dim=1).view([-1, self.num_features, 1, 1]).repeat(1, 1, height, width)
    beta = torch.cat([self.beta for _ in range(batch_size)], dim=1).view([-1, self.num_features, 1, 1]).repeat(1, 1, height, width)
    return normed * gamma + beta