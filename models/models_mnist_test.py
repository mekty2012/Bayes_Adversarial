import models_mnist
import torch


x = torch.rand([2, 1, 28, 28])
m = models_mnist.gaussian_resnet18("sq", dict(), False)
print(m(x).shape)
m = models_mnist.gaussian_resnet18("exp", dict(), False)
print(m(x).shape)
m = models_mnist.gaussian_resnet18("sq", dict(), True)
print(m(x).shape)
m = models_mnist.gaussian_resnet18("exp", dict(), True)
print(m(x).shape)

x = torch.rand([2, 1, 28, 28])
m = models_mnist.dropout_resnet18(0.2, "w", dict())
print(m(x).shape)
m = models_mnist.dropout_resnet18(0.2, "f", dict())
print(m(x).shape)
m = models_mnist.dropout_resnet18(0.2, "c", dict())
print(m(x).shape)

m = models_mnist.ensemble_resnet18(5)
y = m(x)
print(len(y), y[0].shape)

m = models_mnist.batchensemble_resnet18(3)
y = m(x)
print(y.shape)