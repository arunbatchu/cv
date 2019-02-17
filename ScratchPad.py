
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from pylab import *



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = resize_PIL_image(image,resize_to=256)
    image = centercrop_PIL_image(image,crop_size=224)

    #Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np.array(image)
    np_image = np_image / np_image.max()
    np_image = (np_image - mean)/std
    np_image = np_image.transpose(2,0,1)

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


def process_image_old(image):
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
    np_image = img_tensor.numpy()
    print(np_image.max())
    print(np_image.min())
    return np_image


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.axis('off')
    ax.imshow(image)


img_pil=Image.open("flowers/test/10/image_07090.jpg")
# img_pil.show()
imshow(process_image(img_pil))
plt.show()

