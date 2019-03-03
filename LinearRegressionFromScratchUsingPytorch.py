# source: https://www.kaggle.com/aakashns/pytorch-basics-linear-regression-from-scratch/notebook
import torch
import numpy as np

x = torch.tensor(3.) #x is input tensor
weights = torch.tensor(4.,requires_grad=True) # weights tensor in a linear equation such as y = mx + b
bias = torch.tensor(5.,requires_grad=True) # bias tensor in the linear equation

print(x)
print(weights)
print(bias)

#Lets calculate y
y = weights*x + bias
print(y)

#calculate gradient
y.backward()

print('dy/dw:',weights.grad)
print('dy/db:',bias.grad)
