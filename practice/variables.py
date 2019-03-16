import torch
from torch.autograd import Variable



x = Variable(torch.FloatTensor([11.2]), requires_grad=True)
y = 2*x
print (x.data)
print(y.data)

print(x.grad_fn)
print(y.grad_fn)

y.backward()
print (x.grad)
print(y.grad)