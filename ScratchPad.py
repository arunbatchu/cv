# TODO: Define your transforms for the training, validation, and testing sets
import torch
from PIL import Image
from torchvision import transforms, datasets
# training_transforms =  transforms.Compose([transforms.Resize(255),
# #                                        transforms.RandomRotation(30),
# #                                        transforms.RandomResizedCrop(224),
#                                        transforms.CenterCrop(224),
#                                        transforms.RandomHorizontalFlip(),
#                                        transforms.ToTensor(),
#                                        transforms.Normalize([0.485,0.456,0.406],
#                                                             [0.229,0.224,.225])
#                                       ])
# testing_transforms = transforms.Compose([transforms.Resize(255),
#                                       transforms.CenterCrop(224),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize([0.485,0.456,0.406],
#                                                             [0.229,0.224,.225])
#                                       ])
#
#
# # TODO: Load the datasets with ImageFolder
# training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
# testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)
# print(testing_dataset.class_to_idx.items())
#
#
# # TODO: Using the image datasets and the trainforms, define the dataloaders
# training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle = True)
# testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=64)
#
# from PIL import Image
# import torchvision.transforms.functional as TF
#
# image = Image.open('YOUR_PATH')
# x = TF.to_tensor(image)
# x.unsqueeze_(0)
# print(x.shape)
#
# output = model(X)

################3
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

img_pil=Image.open("/home/arun/Downloads/cat.jpg")
img_pil.show()
img_tensor = preprocess(img_pil)
img_tensor.unsqueeze_(0)


#################
def view_classify(img, ps, class_to_idx):
    ''' Function for viewing an image and it's predicted classes.
    '''
    values, indices = ps.topk(5)
    indices = indices.to('cpu')
    print(indices)
    #     print(indices)
    print(values)
    #     labels = [cat_to_name[i] for i in indices.to('cpu')]
    #     print(labels)
    img, ps = img.to('cpu'), ps.to('cpu')
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    ps = ps.data.numpy().squeeze()
    indices_list = indices.tolist()
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    top_5_categories = [cat_to_name[category_key] for category_key in [idx_to_class[index] for index in indices_list]]

    fig, (ax1, ax2) = plt.subplots(figsize=(20, 9), ncols=2)
    ax1.imshow(img)
    ax1.axis('off')
    ax2.barh(np.arange(5), values)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(5))

    ax2.set_yticklabels(top_5_categories)

    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)