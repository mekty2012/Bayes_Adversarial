# Bayes_Adversarial
We study the effect of various BNNs on the adversarial loss.

## TODO

Implement other architectures, for our datasets.
- MNIST
- CIFAR10
- ImageNet
- FGVC

Implement adversarial attack for BNNs.

Implement training of BNNs.
- Gaussian NN
- Dropout NN
- Ensemble
- SWAG


## Explanation for each files

### `layers.py`

This file contains the Bayesian layers, especially the variant of nn.Linear and nn.Conv2D.

All layers have additional parameter `init_dict`, which allows to initialize the weights with different mean/variance. By default, He initialization is used.

#### Gaussian Layers

[Weight Uncertainties in Neural Networks](https://arxiv.org/abs/1505.05424)

- `Gaussian_Linear` : `nn.Linear`
- `Gaussian_Conv2D`, `Gaussian_Conv2D_LRT` : `nn.Conv2D`

These classes requires one new parameter `var_type`, which is the model of using its variance. `"sq"` uses the sigma parameter as std, 
```weight_var = torch.square(weight_sigma)```
and `"exp"` uses the sigma parameter as log std.
```weight_var = torch.exp(weight_sigma)```

The difference between `Gaussian_Conv2D` and `Gaussian_Conv2D_LRT` is that, LRT uses Local Reparameterization Trick ([Variational Dropout and the Local Reparameterization Trick](https://arxiv.org/abs/1506.02557)). LRT is much faster, however it does not exactly follow the weight sharing of convolution.

#### Dropout Layers

[Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)

- `Dropout_Linear` : `nn.Linear`
- `Dropout_Conv2D` : `nn.Conv2D`

These classes requires two new parameter `dropout_rate` and `dropout_type`. The `dropout_rate` is the probability that we drop the entries. The `dropout_type` is the mode of how we dropout the values. `"w"` dropouts the weights, `"f"` dropouts the activations, `"c"` dropouts the channels. `"w"` is slow, but matches the original paper, where `"f"`, `"c"` is suggested in the above paper variational dropout. 

### models.py

Currently, all the implemented models are ResNet18, with input dimension (1, 224, 224). The output has shape (20), 10 for the output, and 10 for the uncertainty.

#### `gaussian_resnet18`

Based on Gaussian Layers. 

#### `dropout_resnet18`

Based on Dropout Layers.

#### `ensemble_resnet18`

(Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles)[https://arxiv.org/abs/1612.01474]

Generate `num_ensemble` numbers of Vanilla Resnet18. Its output is (20) * `num_ensemble`.

#### `get_swag_parameter`, `swag_resnet18_inference`

(A Simple Baseline for Bayesian Uncertainty in Deep Learning)[https://arxiv.org/abs/1902.02476]

First, train the vanilla ResNet18. Then use `get_swag_parameter` to get the parameters for SWAG. You need training details like `train_dataloader`, `lr`, `loss_fn`. 
This function will sample `num_samples` parameters to approximate mean and variance, each parameters are sampled per `frequency` SGD steps. The `rank` adds low rank approximation for covariance. Higher `rank` is slower, but stronger.

Then applying these parameters to `swag_resnet18_inference` gives results. 

### `adversarial.py`

Implement the adversarial attacks.
We may assume that we have `N` samples of output and their uncertainties.

