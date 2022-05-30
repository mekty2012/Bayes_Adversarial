import numpy as np
import torch
import torch.nn

def projected_gradient_descent(x, model, label, target=None):
  """
  x : Image, without minibatch. [1, 3, width, height]
  model : BNN. Can be applied several times. Returns output y and its confidence c.
          [batch, 3, width, height] -> (y:[batch, class_num], c:[batch, class_num])
  label : Index of true label.
  target : The target of adversarial. If none, train to reduce the probability of true label.
           If not none, it is index of label. 
           Train to reduce the probability of true label and increase the probability of target label.
  There may should be several parameters, like learning rate.
  """
  pass

def fast_gradient_sign_method():
  """
  x : Image, without minibatch. [1, 3, width, height]
  model : BNN. Can be applied several times. Returns output y and its confidence c.
          [batch, 3, width, height] -> (y:[batch, class_num], c:[batch, class_num])
  label : Index of true label.
  target : The target of adversarial. If none, train to reduce the probability of true label.
           If not none, it is index of label. 
           Train to reduce the probability of true label and increase the probability of target label.
  There may should be several parameters, like learning rate.
  """
  pass