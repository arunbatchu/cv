from torchvision import transforms
import torch
import helper as h
from PIL import Image
import argparse
def process_image(image):
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


def predict(image_path, model, topk=5, device_type='cuda'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device_type)
    model.eval()
    img_pil=Image.open(image_path)
    img_tensor = process_image(img_pil)
    #TODO convert to plural images. How?
    images = img_tensor.to(device_type).unsqueeze_(0) #convert to device type and add batchsize=1 to first column
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(images)
    probabilities = torch.exp(output)
    probabilities = probabilities.to('cpu')
    # calculate topk
    #probabilities = probabilities.data.numpy().squeeze() #What does squeeze do?
    values, indices = probabilities.topk(topk)
    indices = indices.data.numpy().squeeze()
    indices_list = indices.tolist()
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    cat_to_name = h.getCategoryNamesDictionary()
    topK_classes = [cat_to_name[category_key] for category_key in [idx_to_class[index] for index in indices_list]]
    return topK_classes, values

# Get command line arguments
#
parser =argparse.ArgumentParser()
parser.add_argument("--gpu",help="Optional to run on gpu if available", action='store_true')
parser.add_argument("--top_k", type=int, required=False, help="Top K most likely classes. There are a total of 102 classes.", default=5)
parser.add_argument("--flowerpath", type=str, required=True, help="File path to flower jpeg")
parser.add_argument("--checkpointpath", type=str, default="checkpoint.pth"
                    , help="File path to checkpoint file. Default is checkpoint.pth in current directory."
                    )
namespace = parser.parse_args()

test_path_name = namespace.flowerpath
model = h.retrieveModelFromCheckpoint(namespace.checkpointpath,hidden_units=512,output_units=102)
#By default, choose device type to be cpu
device_type = 'cpu'
if namespace.gpu == True:
    device_type = 'cuda'
topK_classes, probabilities = predict(test_path_name, model, namespace.top_k, device_type=device_type)
probabilities = probabilities.data.numpy().squeeze()#tensor --> numpy and flatten
print("Top {} classes".format(namespace.top_k))
index = 0
for classname in topK_classes:
    index+=1
    print("{}. {:15} with probability {:5.2f}%".format(index,classname,round(100*probabilities[index-1],2)))