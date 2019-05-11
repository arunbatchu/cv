import argparse

import numpy as np
import torch
from PIL import Image
from numpy.core.multiarray import ndarray
from torchvision import transforms

import helper as h


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = resize_PIL_image(image, resize_to=256)
    image = centercrop_PIL_image(image, crop_size=224)

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np.array(image)
    np_image = np_image / np_image.max()
    np_image = (np_image - mean) / std
    np_image = np_image.transpose(2, 0, 1)

    return np_image


def centercrop_PIL_image(image, crop_size=224):
    # Center crop
    left = (image.width - crop_size) / 2
    top = (image.height - crop_size) / 2
    right = (image.width + crop_size) / 2
    bottom = (image.height + crop_size) / 2
    image = image.crop((left, top, right, bottom))
    return image


def resize_PIL_image(image, resize_to=256):
    # Resize
    shortest_side = min(image.width, image.height)
    height = int((image.height / shortest_side) * resize_to)
    width = int((image.width / shortest_side) * resize_to)
    # Resize takes a tuple
    image = image.resize((width, height))
    return image


def process_image_using_pytorch_transforms(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
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
    img_tensor = preprocess(image)

    return img_tensor


def predict(image_path, model, topk=5, device_type='cuda', cat_to_name_file=None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device_type)
    model.eval()
    img_pil = Image.open(image_path)
    np_image: ndarray = process_image(img_pil)
    img_tensor = torch.from_numpy(np_image).float()

    images = img_tensor.to(device_type).unsqueeze_(0)  # convert to device type and add batchsize=1 to first column
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(images)
    probabilities = torch.exp(output)
    probabilities = probabilities.to('cpu')
    # calculate topk
    values, indices = probabilities.topk(topk)
    indices = indices.data.numpy().squeeze()
    indices_list = indices.tolist()
    print(indices_list)
    if (cat_to_name_file != None):
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        cat_to_name = h.getCategoryNamesDictionary(cat_to_name_file)
        topK_classes = [cat_to_name[category_key] for category_key in [idx_to_class[index] for index in indices_list]]
        return topK_classes, values
    else:
        return indices_list, values


#
# Get command line arguments
#
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", help="Optional to run on gpu if available", action='store_true')
parser.add_argument("--top_k", type=int, required=False,
                    help="Top K most likely classes. There are a total of 102 classes.", default=5)
parser.add_argument("--imagepath", type=str, required=True, help="File path to jpeg to be classified")
parser.add_argument("--cat_to_name", type=str, required=False,
                    help="File path to category to name mapping in json format")
parser.add_argument("--checkpointpath", type=str, default="checkpoint.pth"
                    , help="File path to checkpoint file. Default is checkpoint.pth in current directory."
                    )
namespace = parser.parse_args()

test_path_name = namespace.imagepath
model = h.retrieveModelFromCheckpoint(namespace.checkpointpath)
device_type = h.get_device_type(namespace.gpu)
topK_classes, probabilities = predict(test_path_name, model, namespace.top_k, device_type=device_type,
                                      cat_to_name_file=namespace.cat_to_name)
probabilities = probabilities.data.numpy().squeeze()  # tensor --> numpy and flatten
print("Top {} classes".format(namespace.top_k))
index = 0
for classname in topK_classes:
    index += 1
    print("{}. {:15} with probability {:5.2f}%".format(index, classname, round(100 * probabilities[index - 1], 2)))
