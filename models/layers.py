import numpy as np
import torch
import torch.nn as nn
from math import sqrt

eps_device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    
    # Compute the He init value, and set the dict.
    k = sqrt(1 / in_features)
    self.init_dict = {"w_mu_mean" : 0,  "w_mu_std" : k, 
                      "w_sig_mean" : 0, "w_sig_std" : k, 
                      "b_mu_mean" : 0,  "b_mu_std" : k, 
                      "b_sig_mean" : 0, "b_sig_std" : k}
    # Update with input dictionary.
    for k, v in init_dict.items():
      self.init_dict[k] = v
    
    self.use_bias = bias
    
    # Define weight parameters, w_mu, w_sigma.
    self.weight_mu = nn.Parameter(data=torch.normal(self.init_dict["w_mu_mean"], self.init_dict["w_mu_std"], [in_features, out_features], dtype=dtype, device=device))
    self.weight_sigma = nn.Parameter(data=torch.normal(self.init_dict["w_sig_mean"], self.init_dict["w_sig_std"], [in_features, out_features], dtype=dtype, device=device))

    # If bias is True, define bias parameters, b_mu, b_sigma.
    if self.use_bias:
      self.bias_mu = nn.Parameter(data=torch.normal(self.init_dict["b_mu_mean"], self.init_dict["b_mu_std"], [out_features], dtype=dtype, device=device))
      self.bias_sigma = nn.Parameter(data=torch.normal(self.init_dict["b_sig_mean"], self.init_dict["b_sig_std"], [out_features], dtype=dtype, device=device))
    
    # Check the variance type.
    if var_type != "exp" and var_type != "sq":
      raise ValueError("The variance mode should be exp or sq.")
    self.var_type = var_type
    
  def forward(self, x):
    if self.use_bias: # Bias is True
      # Compute mean of output, m = W_mu x + b_mu
      new_mean = torch.matmul(x, self.weight_mu) + self.bias_mu # [batch_size, out_features]
      # Compute variance of output, v = W_var X + b_var
      if self.var_type == "exp":
        new_var = torch.matmul(torch.square(x), torch.exp(self.weight_sigma)) + torch.exp(self.bias_sigma) # [batch_size, out_features]
      elif self.var_type == "sq":
        new_var = torch.matmul(torch.square(x), torch.square(self.weight_sigma)) + torch.square(self.bias_sigma) # [batch_size, out_features]
    else:
      # Compute mean of output, m = W_mu x + b_mu
      new_mean = torch.matmul(x, self.weight_mu) # [batch_size, out_features]
      # Compute variance of output, v = W_var X + b_var
      if self.var_type == "exp":
        new_var = torch.matmul(torch.square(x), torch.exp(self.weight_sigma)) # [batch_size, out_features]
      elif self.var_type == "sq":
        new_var = torch.matmul(torch.square(x), torch.square(self.weight_sigma)) # [batch_size, out_features]
    
    # Reparameterization
    eps = torch.normal(0, 1, new_mean.shape, device=eps_device) # [batch_size, out_features]
    return new_mean + eps * torch.sqrt(new_var) # [batch_size, out_features]

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
               in_channels, # Number of input channel. Same as Conv2D.
               out_channels, # Number of output channel. Same as Conv2D.
               kernel_size, # Kernel size. Same as Conv2D.
               var_type, # Type of variance mode.
               stride=1, # Stride size. Same as Conv2D.
               padding=0, # Padding size. Same as Conv2D. padding mode is not supported.
               dilation=1, # Dilation size. Same as Conv2D.
               groups=1, # Group size. Same as Conv2D.
               bias=True, # Use bias or not. Same as Conv2D.
               device=None, 
               dtype=None,
               init_dict=dict() # The dictionary of parameters.
               ):
    super(Gaussian_Conv2D_LRT, self).__init__()
    
    # Stores the parameters.
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

    # Check the variance type.
    if var_type != "exp" and var_type != "sq":
      raise ValueError("The variance mode should be exp or sq.")
    self.var_type = var_type

    # Compute the He init parameter.
    k = sqrt(groups / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
    self.init_dict = {"w_mu_mean" : 0,  "w_mu_std" : k, 
                      "w_sig_mean" : 0, "w_sig_std" : k, 
                      "b_mu_mean" : 0,  "b_mu_std" : k, 
                      "b_sig_mean" : 0, "b_sig_std" : k}
    # Update by the input dict.
    for k, v in init_dict.items():
      self.init_dict[k] = v
    
    # Check the group divisibility.
    if self.in_channels % self.groups != 0:
      raise ValueError("in_channels must be divisible by groups")
    
    # Define the weight parameters, w_mu, w_sigma.
    shape = [self.out_channels, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]]
    self.weight_mu = nn.Parameter(data=torch.normal(self.init_dict["w_mu_mean"], self.init_dict["w_mu_std"], shape, dtype=dtype, device=device))
    self.weight_sigma = nn.Parameter(data=torch.normal(self.init_dict["w_sig_mean"], self.init_dict["w_sig_std"], shape, dtype=dtype, device=device))
    
    # If use bias, define the bias parameters, b_mu, b_sigma.
    if self.use_bias:
      self.bias_mu = nn.Parameter(data=torch.normal(self.init_dict["b_mu_mean"], self.init_dict["b_mu_std"], [self.out_channels], dtype=dtype, device=device))
      self.bias_sigma = nn.Parameter(data=torch.normal(self.init_dict["b_sig_mean"], self.init_dict["b_sig_std"], [self.out_channels], dtype=dtype, device=device))
    

  def forward(self, x):
    if self.use_bias:
      # Define the output mean, m = conv(x ; W_mu) + b_mu.
      new_mean = nn.functional.conv2d(x, self.weight_mu, self.bias_mu, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) # [batch_size, out_channels, new_height, new_width]
      # Define the output variance, V = conv(x ; W_var) + b_var
      if self.var_type == "exp":
        new_var = nn.functional.conv2d(torch.square(x), torch.exp(self.weight_sigma), torch.exp(self.bias_mu), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) # [batch_size, out_channels, new_height, new_width]
      else:
        new_var = nn.functional.conv2d(torch.square(x), torch.square(self.weight_sigma), torch.square(self.bias_sigma), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) # [batch_size, out_channels, new_height, new_width]
    else:
      # Define the output mean, m = conv(x ; W_mu) + b_mu.
      new_mean = nn.funcitonal.conv2d(x, self.weight_mu, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) # [batch_size, out_channels, new_height, new_width]
      # Define the output variance, V = conv(x ; W_var) + b_var
      if self.var_type == "exp":
        new_var = nn.functional.conv2d(torch.square(x), torch.exp(self.weight_sigma), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) # [batch_size, out_channels, new_height, new_width]
      else:
        new_var = nn.functional.conv2d(torch.square(x), torch.square(self.weight_sigma), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) # [batch_size, out_channels, new_height, new_width]

    # (Local) Reparameterization trick
    eps = torch.normal(0, 1, new_mean.shape, device=eps_device) # [batch_size, out_channels, new_height, new_width]
    return new_mean + eps * torch.sqrt(new_var) # [batch_size, out_channels, new_height, new_width]

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
               in_channels, # Number of input channel. Same as Conv2D.
               out_channels, # Number of output channel. Same as Conv2D.
               kernel_size, # Kernel size. Same as Conv2D.
               var_type, # Type of variance mode.
               stride=1, # Stride size. Same as Conv2D.
               padding=0, # Padding size. Same as Conv2D. padding mode is not supported.
               dilation=1, # Dilation size. Same as Conv2D.
               groups=1, # Group size. Same as Conv2D.
               bias=True, # Use bias or not. Same as Conv2D.
               device=None,
               dtype=None,
               init_dict=dict() # The dictionary of parameters.
               ):
    super(Gaussian_Conv2D, self).__init__()

    # Stores the parameters.
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
    
    # Compute the He init parameter.
    k = sqrt(groups / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
    self.init_dict = {"w_mu_mean" : 0,  "w_mu_std" : k, 
                      "w_sig_mean" : 0, "w_sig_std" : k, 
                      "b_mu_mean" : 0,  "b_mu_std" : k, 
                      "b_sig_mean" : 0, "b_sig_std" : k}
    # Update by the input dict.
    for k, v in init_dict.items():
      self.init_dict[k] = v
    
    # Check the group divisibility.
    if self.in_channels % self.groups != 0:
      raise ValueError("in_channels must be divisible by groups")
    
    # Define the weight parameters, w_mu, w_sigma.
    shape = [self.out_channels, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]]
    self.weight_mu = nn.Parameter(data=torch.normal(self.init_dict["w_mu_mean"], self.init_dict["w_mu_std"], shape, dtype=dtype, device=device))
    self.weight_sigma = nn.Parameter(data=torch.normal(self.init_dict["w_sig_mean"], self.init_dict["w_sig_std"], shape, dtype=dtype, device=device))
    
    # If use bias, define the bias parameters, b_mu, b_sigma.
    if self.use_bias:
      self.bias_mu = nn.Parameter(data=torch.normal(self.init_dict["b_mu_mean"], self.init_dict["b_mu_std"], [self.out_channels], dtype=dtype, device=device))
      self.bias_sigma = nn.Parameter(data=torch.normal(self.init_dict["b_sig_mean"], self.init_dict["b_sig_std"], [self.out_channels], dtype=dtype, device=device))
    
  def forward(self, x):
    # This time, we iterate for each samples in minibatch.
    batch_size = x.shape[0]
    res = []

    # Pre-compute the std of weight.
    if self.var_type == "exp":
      weight_std = torch.exp(self.weight_sigma / 2)
      if self.use_bias:
        bias_std = torch.exp(self.bias_sigma / 2)
    else:
      weight_std = self.weight_sigma
      if self.use_bias:
        bias_std = self.bias_sigma
    for i in range(batch_size):
      # Get i-th input.
      xi = torch.unsqueeze(x[i, :, :, :], dim=0) # [1, channel, height, width]
      if self.use_bias:
        # Sample weights, with reparameterization.
        w_eps = torch.normal(0, 1, self.weight_mu.shape, device=eps_device)
        b_eps = torch.normal(0, 1, self.bias_mu.shape, device=eps_device)
        new_weight = self.weight_mu + w_eps * torch.abs(weight_std)
        new_bias = self.bias_mu + b_eps * torch.abs(bias_std)
        
        # Compute the i-th output, and store.
        yi = nn.functional.conv2d(xi, new_weight, new_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) # [1, out_channel, new_height, new_width]
        res.append(yi)
      else:
        # Sample weights, with reparameterization.
        new_weight = torch.normal(self.weight_mu, torch.abs(weight_std))
        # Compute the i-th output, and store.
        yi = nn.functional.conv2d(xi, new_weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        res.append(yi) # [1, out_channel, new_height, new_width]
    return torch.concat(res, dim=0) # [batch_size, out_channel, new_height, new_width]

class Gaussian_BatchNorm2D(nn.Module):
  """
  Implements the BatchNorm2D for BNN.
  The gamma and beta parameters are Gaussians now.
  """
  def __init__(self,
               num_features, # Number of features.
               var_type, # Type of variances.
               eps=1e-05, # Epsilon to prevent the numerical error. Same as BatchNorm2D.
               momentum=0.1, # Momentum of tracking the statistics. Same as BatchNorm2D.
               track_running_stats=True, # Whether to track the stats or not. Same as BatchNorm2D.
               device=None,
               dtype=None,
               init_dict=dict() # The dictionary of parameters.
               ):
    super(Gaussian_BatchNorm2D, self).__init__()
    
    # Define its internal BatchNorm. It do not contain gamma, beta parameters.
    self.batch_norm = nn.BatchNorm2d(num_features, eps, momentum, affine=False, track_running_stats=track_running_stats, device=device, dtype=dtype)
    
    # Define the init parameters.
    self.init_dict = {"gamma_mu_mean" : 1, "gamma_mu_std" : 0.1,
                      "gamma_sigma_mean" : 0, "gamma_sigma_std" : 1,
                      "beta_mu_mean" : 0, "beta_mu_std" : 0.1,
                      "beta_sigma_mean" : 0, "beta_sigma_std" : 1}
    for k, v in init_dict.items():
      self.init_dict[k] = v
    
    # Define gamma parameters, gamma_mu, gamma_sigma. The '1's are added to match the dimension of input.
    self.gamma_mu = nn.Parameter(torch.normal(self.init_dict["gamma_mu_mean"], self.init_dict["gamma_mu_std"], [1, num_features, 1, 1], device=device, dtype=dtype))
    self.gamma_sigma = nn.Parameter(torch.normal(self.init_dict["gamma_sigma_mean"], self.init_dict["gamma_sigma_std"], [1, num_features, 1, 1], device=device, dtype=dtype))
    
    # Define beta_parameters, beta_mu, beta_sigma. 
    self.beta_mu = nn.Parameter(torch.normal(self.init_dict["beta_mu_mean"], self.init_dict["beta_mu_std"], [1, num_features, 1, 1], device=device, dtype=dtype))
    self.beta_sigma = nn.Parameter(torch.normal(self.init_dict["beta_sigma_mean"], self.init_dict["beta_sigma_std"], [1, num_features, 1, 1], device=device, dtype=dtype))
    
    # Check the variance type.
    if var_type != "exp" and var_type != "sq":
      raise ValueError("The variance mode should be exp or sq.")
    self.var_type = var_type
  
  def forward(self, x):
    # First get the dimension.
    batch_size = x.shape[0]
    num_channel = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    
    # Repeat the parameters to match the size of input.
    gamma_mu = self.gamma_mu.repeat([batch_size, 1, height, width])
    gamma_sigma = self.gamma_sigma.repeat([batch_size, 1, height, width])
    beta_mu = self.beta_mu.repeat([batch_size, 1, height, width])
    beta_sigma = self.beta_sigma.repeat([batch_size, 1, height, width])
    
    # Compute the mean, batch_norm(x) * gamma_mu + beta_mu.
    normed = self.batch_norm(x) # [batch_size, channels, height, width]
    new_mean = normed * gamma_mu + beta_mu # [batch_size, channels, height, width]
    
    # Compute the variance. batch_norm(x)^2 * gamma_var + beta_var.
    if self.var_type == "sq":
      new_var = torch.square(normed) * torch.square(gamma_sigma) + torch.square(beta_sigma)
    else:
      new_var = torch.square(normed) * torch.exp(gamma_sigma) + torch.exp(beta_sigma)

    # Compute the output, by reparameterization trick.
    eps = torch.normal(0, 1, new_mean.shape, device=eps_device)
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
               in_features, # Input dimension 
               out_features, # Output dimension
               dropout_rate, # The dropout rate. 
               dropout_type,# Type of dropout. 
               bias=True, # Use bias or not.
               device=None,
               dtype=None,
               init_dict=dict() # The dictionary of parameters.
               ):
    super(Dropout_Linear, self).__init__()
    
    # Compute the He init value, and set the dict.
    k = sqrt(1 / in_features)
    self.init_dict = {"w_mean" : 0, "w_std" : k,
                      "b_mean" : 0, "b_std" : k}
    # Update with input dictionary.
    for k, v in init_dict.items():
      self.init_dict[k] = v

    self.use_bias = bias

    # Define weight parameter.
    self.weight = nn.Parameter(data=torch.normal(self.init_dict["w_mean"], self.init_dict["w_std"], [in_features, out_features], dtype=dtype, device=device))
    
    # If bias is True, define bias parameters, b_mu, b_sigma.
    if self.use_bias:
      self.bias = nn.Parameter(data=torch.normal(self.init_dict["b_mean"], self.init_dict["b_std"], [out_features], dtype=dtype, device=device))
    
    # Store dropout parameter. 'c' will be translated to 'f'.
    self.dropout_rate = dropout_rate
    if dropout_type not in ["w", "f", "c"]:
      raise ValueError("dropout_type should be either w(weight), f(feature), c(channel).")
    self.dropout_type=dropout_type
  
  def dropout(self, x):
    return x * (torch.rand(x.shape, device=eps_device) > self.dropout_rate)
  
  def forward(self, x):
    if self.dropout_type == "w":
      # We drop the weights. We will compute it for each elements in batch.
      res = []
      batch_size = x.shape[0]
      for i in range(batch_size):
        # i-th input.
        xi = torch.unsqueeze(x[i, :], 0) # [1, in_features]
        
        if self.use_bias:
          # Dropout weights, and compute. 
          # y = dropout(W) * x + dropout(b).
          new_w = self.dropout(self.weight)
          new_b = self.dropout(self.bias)
          yi = torch.matmul(xi, new_w) + new_b # [1, out_features]
          res.append(yi)
        else:
          # Dropout weights, and compute. 
          # y = dropout(W) * x.
          new_w = self.dropout(self.weight)
          yi = torch.matmul(xi, new_w) # [1, out_features]
          res.append(yi)
      return torch.concat(res, dim=0) # [batch_size, out_features]
    else:
      # We drop the features. Simply drop the outputs.
      if self.use_bias:
        output = torch.matmul(x, self.weight) + self.bias
        return self.dropout(output)
      else:
        output = torch.matmul(x, self.weight)
        return self.dropout(output)

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
               in_channels, # Number of input channel. Same as Conv2D.
               out_channels, # Number of output channel. Same as Conv2D.
               kernel_size, # Kernel size. Same as Conv2D.
               dropout_rate, # Dropout Rate.
               dropout_type, # Mode of dropout.
               stride=1, # Stride size. Same as Conv2D.
               padding=0, # Padding size. Same as Conv2D. padding mode is not supported.
               dilation=1, # Dilation size. Same as Conv2D.
               groups=1, # Group size. Same as Conv2D.
               bias=True, # Use bias or not. Same as Conv2D.
               device=None, 
               dtype=None,
               init_dict=dict() # The dictionary of parameters.
               ):
    super(Dropout_Conv2D, self).__init__()
    
    # Stores the parameters.
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
    self.dropout_rate = dropout_rate

    # Check the variance type.
    if dropout_type not in ["w", "f", "c"]:
      raise ValueError("dropout_type should be either w(weight), f(feature), c(channel).")
    self.dropout_type = dropout_type
    
    # Compute the He init parameters.
    k = sqrt(groups / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
    # Update by the input dict.
    self.init_dict = {"w_mean" : 0, "w_std" : k,
                      "b_mean" : 0, "b_std" : k}
    for k, v in init_dict.items():
      self.init_dict[k] = v
    
    # Check the group divisibility.
    if self.in_channels % self.groups != 0:
      raise ValueError("in_channels must be divisible by groups")
    
    # Define the weight parameter
    shape = [self.out_channels, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]]
    self.weight = nn.Parameter(data=torch.normal(self.init_dict["w_mean"], self.init_dict["w_std"], shape, dtype=dtype, device=device))
    
    # If use bias, define the bias parameter.
    if self.use_bias:
      self.bias = nn.Parameter(data=torch.normal(self.init_dict["b_mean"], self.init_dict["b_std"], [out_channels], dtype=dtype, device=device))

  def dropout(self, x):
    return x * (torch.rand(x.shape, device=eps_device) > self.dropout_rate)

  def forward(self, x):
    if self.dropout_type == "w":
      # We drop the weights. We will compute it for each elements in batch.
      res = []
      batch_size = x.shape[0]
      for i in range(batch_size):
        # i-th input.
        xi = torch.unsqueeze(x[i, :, :, :], 0) # [1, in_channels, height, width]
        if self.use_bias:
          # Dropout weights, and compute. 
          # y = conv(x ; dropout(W)) + dropout(b).
          new_w = self.dropout(self.weight)
          new_b = self.dropout(self.bias)
          yi = nn.functional.conv2d(xi, new_w, new_b, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) # [1, out_channels, new_height, new_width]
          res.append(yi)
        else:
          # Dropout weights, and compute. 
          # y = conv(x ; dropout(W)).
          new_w = self.dropout(self.weight)
          yi = nn.functional.conv2d(xi, new_w, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) # [1, out_channels, new_height, new_width]
          res.append(yi)
      return torch.concat(res, dim=0) # [batch_size, out_channels, new_height, new_width]
    elif self.dropout_type == "f":
      # We dropout the features, without sharing between channels or pixel positions.
      if self.use_bias:
        # Compute the output.
        # y = conv(x; W) + b
        output = nn.functional.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
      else:
        # Compute the output.
        # y = conv(x; W)
        output = nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
      # Compute the dropout.
      return self.dropout(output)
    else:
      # We dropout the channels, so all pixels in same channels are dropped together.
      if self.use_bias:
        # Compute the output.
        # y = conv(x; W) + b
        output = nn.functional.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
      else:
        # Compute the output.
        # y = conv(x; W)
        output = nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
      # Dropout by channel dropout.
      return nn.functional.dropout2d(output, self.dropout_rate)

class Dropout_BatchNorm2D(nn.Module):
  """
  Implements the BatchNorm2D for MC-dropout.
  Gamma, Beta can be dropouted, except that, it is equivalent to nn.BatchNorm2D.

  dropout_rate : The rate of dropout. When 0, no dropout. 
  dropout_type : Mode of dropout.
                 "w" : Dropout the weight. It makes computation iteration over batch (slow).
                 "f" : Dropout the output feature. Faster, and is equivalent to dropping row-wise.
                 "c" : Dropout the output channel. Faster, and is equivalent to droppint all channel weight.
  """
  def __init__(self,
               num_features, # Number of features.
               dropout_rate, # Dropout rate.
               dropout_type, # Mode of dropout.
               eps=1e-05, # Epsilon to prevent the numerical error. Same as BatchNorm2D.
               momentum=0.1, # Momentum of tracking the statistics. Same as BatchNorm2D.
               track_running_stats=True, # Whether to track the stats or not. Same as BatchNorm2D.
               device=None,
               dtype=None,
               ):
    super(Dropout_BatchNorm2D, self).__init__()
    
    # Define its internal BatchNorm. It do not contain gamma, beta parameters.
    self.batch_norm = nn.BatchNorm2d(num_features, eps, momentum, affine=False, track_running_stats=track_running_stats, device=device, dtype=dtype)

    # Define gamma parameter. The '1's are added to match the dimension of input. Init by gamma=1.
    self.gamma = nn.Parameter(torch.ones([1, num_features, 1, 1], device=device, dtype=dtype))
    # Define beta_parameter. Init by beta=0. 
    self.beta = nn.Parameter(torch.zeros([1, num_features, 1, 1], device=device, dtype=dtype))

    if dropout_type not in ["w", "f", "c"]:
      raise ValueError("dropout_type should be either w(weight), f(feature), c(channel).")
    self.dropout_type=dropout_type
    self.dropout_rate = dropout_rate
  
  def dropout(self, x):
    return x * (torch.rand(x.shape, device=eps_device) > self.dropout_rate)

  def forward(self, x):
    # First get the dimension.
    batch_size = x.shape[0]
    num_channel = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]

    # Compute with internal batch norm.
    normed = self.batch_norm(x)
    if self.dropout_type == "w":
      # We drop the weights. We should compute for each element in batch.
      res = []
      # Repeat the parameters to match the size of input.
      gamma = self.gamma.repeat([1, 1, height, width])
      beta = self.beta.repeat([1, 1, height, width])
      for i in range(batch_size):
        xi = torch.unsqueeze(normed[i, :, :, :], 0) # [1, channels, height, width]
        yi = xi * self.dropout(gamma) + self.dropout(beta) # [1, channels, height, width]
        res.append(yi)
      return torch.concat(res, 0) # [batch_size, channels, height, width]
    else:
      # Repeat the parameters to match the size of input.
      gamma = self.gamma.repeat([batch_size, 1, height, width])
      beta = self.beta.repeat([batch_size, 1, height, width])
      y = normed * gamma + beta # [batch_size, channels, height, width]
      return self.dropout(y) # [batch_size, channels, height, width]

class BatchEnsemble_Linear(nn.Module):
  """
  Implements the Linear layer for batch ensemble.
  W contains two additional parameter alpha, gamma, for each models, so we have each models have weight
  W * (alpha gamma^T).
  We use ensemble for bias.

  num_models : Number of ensemble models. 
  init_dict : Initialization mean/variances for the parameters. By default, He init.
  """
  def __init__(self, 
               in_features,  # Input dimension 
               out_features,  # Output dimension
               num_models, # Number of ensemble models.
               is_first, # True if its the first layer of model.
               bias=True, # True if use bias.
               device=None, 
               dtype=None,
               init_dict=dict(), # The dictionary of parameters.
               ):
    super(BatchEnsemble_Linear, self).__init__()
    
    k = sqrt(1 / in_features)
    self.init_dict = {"w_mean" : 0, "w_std" : k,
                      "b_mean" : 0, "b_std" : k,
                      "alpha_mean" : 0, "alpha_std" : 1,
                      "gamma_mean" : 0, "gamma_std" : 1}
    # Update with input dictionary.
    for k, v in init_dict.items():
      self.init_dict[k] = v

    # Define internal Linear layer, without bias.
    self.fc = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
    nn.init.normal_(self.fc.weight, mean=self.init_dict["w_mean"], std=self.init_dict["w_std"])

    # Define batch ensemble parameters.
    self.alpha = nn.Parameter(torch.normal(self.init_dict["alpha_mean"], self.init_dict["alpha_std"], [num_models, in_features], device=device, dtype=dtype))
    self.gamma = nn.Parameter(torch.normal(self.init_dict["gamma_mean"], self.init_dict["gamma_std"], [num_models, out_features], device=device, dtype=dtype))
    self.use_bias = bias
    if self.use_bias:
      self.bias = nn.Parameter(torch.normal(self.init_dict["b_mean"], self.init_dict["b_std"], [num_models, out_features], device=device, dtype=dtype))
    
    # Store parameters
    self.is_first = is_first
    self.num_models = num_models
    self.in_features = in_features
    self.out_features = out_features

  def forward(self, x):
    # If it is first layer, duplicate the input to [batch_size * num_models, in_features]
    # The form is [x1, x2, ..., xN, x1, x2, ..., xN, ..., x1, x2, ..., xN]
    if self.is_first:
      x = torch.cat([x for _ in range(self.num_models)], dim=0)
    
    # Get the 'real' batch_size
    batch_size = x.shape[0] // self.num_models
    
    # Repeat alpha, beta, bias.
    # The form is [a1, a1, ..., a1, a2, a2, ..., a2, ..., ad, ad, ..., ad]
    alpha = torch.cat([self.alpha for _ in range(batch_size)], dim=1).view(-1, self.in_features) # [batch_size * num_models, in_features]
    gamma = torch.cat([self.gamma for _ in range(batch_size)], dim=1).view(-1, self.out_features) # [batch_size * num_models, out_features]
    if self.use_bias:
      bias = torch.cat([self.bias for _ in range(batch_size)], dim=1).view(-1, self.out_features) # [batch_size * num_models, out_features]
      # Return output.
      # y_i = W * (x * alpha_i) * gamma_i + b_i
      return self.fc(x * alpha) * gamma + bias # [batch_size * num_models, out_features]
    else:
      # Return output.
      # y_i = W * (x * alpha_i) * gamma_i
      return self.fc(x * alpha) * gamma # [batch_size * num_models, out_features]

class BatchEnsemble_Conv2D(nn.Module):
  """
  Implements the Conv2D layer for batch ensemble.
  W contains two additional parameter alpha, gamma, for each models, so we have each models have weight
  W * (alpha gamma^T).
  We use ensemble for bias.

  num_models : Number of ensemble models. 
  init_dict : Initialization mean/variances for the parameters. By default, He init.
  """
  def __init__(self,
               in_channels, # Number of input channel. Same as Conv2D.
               out_channels, # Number of output channel. Same as Conv2D.
               kernel_size, # Kernel size. Same as Conv2D.
               num_models, # Number of ensemble models.
               is_first, # True if its the first layer of model.
               stride=1, # Stride size. Same as Conv2D.
               padding=0, # Padding size. Same as Conv2D. padding mode is not supported.
               dilation=1, # Dilation size. Same as Conv2D.
               groups=1, # Group size. Same as Conv2D.
               bias=True, # Use bias or not. Same as Conv2D.
               device=None, 
               dtype=None,
               init_dict = dict() # The dictionary of parameters.
               ):
    super().__init__()
    
    # Store the parameters
    self.is_first = is_first
    self.use_bias = bias
    self.num_models = num_models
    self.in_channels = in_channels
    self.out_channels = out_channels
    if isinstance(kernel_size, tuple) and len(kernel_size) >= 2:
      self.kernel_size = kernel_size
    elif isinstance(kernel_size, tuple):
      self.kernel_size = (kernel_size[0], kernel_size[0])
    else:
      self.kernel_size = (kernel_size, kernel_size)

    # Define internal conv2d layer.
    self.conv2d = nn.Conv2d(in_channels, out_channels, self.kernel_size, stride, padding, dilation, groups, bias=False, padding_mode="zeros", device=device, dtype=dtype)

    # Compute the He init parameters.
    k = sqrt(groups / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
    self.init_dict = {"w_mean" : 0, "w_std" : k,
                      "b_mean" : 0, "b_std" : k,
                      "alpha_mean" : 0, "alpha_std" : 1,
                      "gamma_mean" : 0, "gamma_std" : 1}
    for k, v in init_dict.items():
      self.init_dict[k] = v
    
    # Reset the conv2d weight.
    nn.init.normal_(self.conv2d.weight, self.init_dict["w_mean"], self.init_dict["w_std"])

    # Define batch ensemble parameters.
    self.alpha = nn.Parameter(torch.normal(self.init_dict["alpha_mean"], self.init_dict["alpha_std"], [num_models, in_channels], device=device, dtype=dtype))
    self.gamma = nn.Parameter(torch.normal(self.init_dict["gamma_mean"], self.init_dict["gamma_std"], [num_models, out_channels], device=device, dtype=dtype))
    if self.use_bias:
      self.bias = nn.Parameter(torch.normal(self.init_dict["b_mean"], self.init_dict["b_std"], [num_models, out_channels], device=device, dtype=dtype))
  
  def forward(self, x):
    # If it is first layer, duplicate the input to [batch_size * num_models, in_features]
    # The form is [x1, x2, ..., xN, x1, x2, ..., xN, ..., x1, x2, ..., xN]
    if self.is_first:
      x = torch.cat([x for _ in range(self.num_models)], dim=0)

    batch_size = x.shape[0] // self.num_models
    height = x.shape[2]
    width = x.shape[3]
    
    # Repeat alpha.
    # The form is [a1, a1, ..., a1, a2, a2, ..., a2, ..., ad, ad, ..., ad]
    alpha = torch.cat([self.alpha for _ in range(batch_size)], dim=1).view([-1, self.in_channels, 1, 1]).repeat(1, 1, height, width)

    # Compute the result of conv2d.
    x = self.conv2d(x * alpha)

    # Get new height and width after convolution.
    new_height = x.shape[2]
    new_width = x.shape[3]

    # Repeat gamma and bias with new height and width.
    gamma = torch.cat([self.gamma for _ in range(batch_size)], dim=1).view([-1, self.out_channels, 1, 1]).repeat(1, 1, new_height, new_width)
    if self.use_bias:
      bias = torch.cat([self.bias for _ in range(batch_size)], dim=1).view([-1, self.out_channels, 1, 1]).repeat(1, 1, new_height, new_width)
      # Return output.
      # y_i = conv2d(x * alpha_i, W) * gamma_i + beta_i
      return x * gamma + bias # [batch_size * num_models, out_channels, new_height, new_widths]
    else:
      # Return output.
      # y_i = conv2d(x * alpha_i, W) * gamma_i
      return x * gamma # [batch_size * num_models, out_channels, new_height, new_widths]

class BatchEnsemble_BatchNorm2D(nn.Module):
  """
  Implements the BatchNorm2D layer for batch ensemble.
  We use naive ensemble for here.

  num_models : Number of ensemble models. 
  """
  def __init__(self,
               num_features, # Number of features.
               num_models, # Number of ensemble models.
               eps=1e-05, # Epsilon to prevent the numerical error. Same as BatchNorm2D.
               momentum=0.1, # Momentum of tracking the statistics. Same as BatchNorm2D.
               track_running_stats=True, # Whether to track the stats or not. Same as BatchNorm2D.
               device=None,
               dtype=None,
               ):
    super().__init__()

    # Define its internal BatchNorm. It do not contain gamma, beta parameters.
    self.batch_norm = nn.BatchNorm2d(num_features, eps, momentum, affine=False, track_running_stats=track_running_stats, device=device, dtype=dtype)

    # Store the parameters.
    self.num_models = num_models
    self.num_features = num_features
    
    # Define parameters. We use initialization of gamma=1, beta=0.
    self.gamma = nn.Parameter(torch.ones([self.num_models, self.num_features], device=device, dtype=dtype))
    self.beta = nn.Parameter(torch.zeros([self.num_models, self.num_features], device=device, dtype=dtype))
    
  def forward(self, x):
    # First get the dimension.
    batch_size = x.shape[0] // self.num_models
    height = x.shape[2]
    width = x.shape[3]  
    
    # Compute with internal batch norm.
    normed = self.batch_norm(x)
    
    # Repeat the gamma and beta to match the size of input.
    # The form is [a1, a1, ..., a1, a2, a2, ..., a2, ..., ad, ad, ..., ad]
    gamma = torch.cat([self.gamma for _ in range(batch_size)], dim=1).view([-1, self.num_features, 1, 1]).repeat(1, 1, height, width)
    beta = torch.cat([self.beta for _ in range(batch_size)], dim=1).view([-1, self.num_features, 1, 1]).repeat(1, 1, height, width)
    # Return y_i = batchnorm(x) * gamma_i + beta_i.
    return normed * gamma + beta