import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, .5, .5), (.5, .5, .5))]
)

training_dataset = torchvision.datasets.CIFAR10(root='./data', train=True
                                                , download=True, transform=transform)
print(len(training_dataset.train_data))
plt.imshow(training_dataset.train_data[1])
print(training_dataset.train_labels[1])

training_dataset_loader = DataLoader(training_dataset
                                     , batch_size=10
                                     , shuffle=True, num_workers=2
                                     )
for i, batch in enumerate(training_dataset_loader):
    data, labels = batch
    print(i)
    print("type(data):",type(data))
    print("data.size():", data.size())
    break
