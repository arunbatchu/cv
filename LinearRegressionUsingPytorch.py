# source: https://www.kaggle.com/aakashns/pytorch-basics-linear-regression-from-scratch/notebook
import torch
import numpy as np
import torch.nn as nn

# Input (temp, rainfall, humidity)
from torch.utils.data import TensorDataset, DataLoader

inputs = np.array(
    [[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58],
     [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]],
    dtype='float32')
# Targets (apples, oranges) yield in tons
targets = np.array([[56, 70], [81, 101], [119, 133], [22, 37], [103, 119],
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119],
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119]], dtype='float32')

# yeild_apple  = w11 * temp + w12 * rainfall + w13 * humidity + b1
# yeild_orange = w21 * temp + w22 * rainfall + w23 * humidity + b2

# Problem statement is : Find suitable weights and bias to satisfy predictions in the training set given above

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# print(inputs)
# print(targets)
#
#Use DataSets and DataLoader

train_ds = TensorDataset(inputs, targets)
# print(type(train_ds[0:3])) # returns a tuple of 2

#Now that we have a data set , lets define a data loader
train_dl = DataLoader(train_ds,batch_size=5, shuffle=True)
train_dl_iterator = iter(train_dl)
# print(next(train_dl_iterator))

def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)

            loss.backward()
            opt.step()
            #zero out the grad
            opt.zero_grad()
    print('Training loss:', loss_fn(model(inputs), targets)) #See how well we are doing after each epoch



# #Lets start with a linear model
# model = nn.Linear(3,2)
# # print(model.weight, model.bias) #show the random numbers chosen
#
# #Optimizer
# opt = torch.optim.SGD(model.parameters(), lr=1e-5)
#
# #loss function
# loss_fn = torch.nn.functional.mse_loss
# loss = loss_fn(model(inputs), targets)
# print(loss)


# fit(500,model,loss_fn,opt)
#
# #generate predictions
# preds = model(inputs)
# print(preds)
# #compare with targets
# print(targets)
# print('###########################')
#
#  Now use a simple feed forward model and see how it works
#
class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3,3)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(3,3)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(3,2)
    def forward(self,x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        return x

model = SimpleNet()
opt = torch.optim.SGD(model.parameters(), lr=.00001)
loss_fn = torch.nn.functional.mse_loss

fit(200,model,loss_fn,opt)

preds = model(inputs)
print(preds)
print(targets)

