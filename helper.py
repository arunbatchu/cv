import json
from datetime import datetime

import torch
from torch import nn, optim
from torchvision import transforms, datasets, models

batch_size = 64


def createNormalizTransform():
    return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, .225])


def createResizeTransform():
    return transforms.Resize(255)


def createCenterCropTransform():
    return transforms.CenterCrop(224)


def createTrainingDataset(data_dir):
    return datasets.ImageFolder(data_dir + '/train'
                                , transform=(transforms.Compose([(createResizeTransform()),
                                                                 transforms.RandomRotation(
                                                                     30),
                                                                 transforms.RandomResizedCrop(
                                                                     224),
                                                                 transforms.RandomHorizontalFlip(),
                                                                 transforms.ToTensor(),
                                                                 (createNormalizTransform())
                                                                 ])))


def createTrainingDataloader(data_dir):
    return torch.utils.data.DataLoader(createTrainingDataset(data_dir), batch_size, shuffle=True)


def createTestingDataset(data_dir):
    return datasets.ImageFolder(data_dir + '/test', transform=(transforms.Compose([(createResizeTransform()),
                                                                                   (createCenterCropTransform()),
                                                                                   transforms.ToTensor(),
                                                                                   (createNormalizTransform())
                                                                                   ])))


def createTestingDataloader(data_dir):
    return torch.utils.data.DataLoader(createTestingDataset(data_dir), batch_size)


def createValidationDataloader(data_dir):
    return torch.utils.data.DataLoader(createValidationDataset(data_dir), batch_size)


def createValidationDataset(data_dir):
    return datasets.ImageFolder(data_dir + '/valid', transforms.Compose([(createResizeTransform()),
                                                                         (createCenterCropTransform()),
                                                                         transforms.ToTensor(),
                                                                         (createNormalizTransform())
                                                                         ]))


def freezeParameters(model):
    for param in model.parameters():
        param.requires_grad = False


def createClassifier(hidden_units, output_units):
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 1000)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(1000, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc3', nn.Linear(hidden_units, output_units)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    classifier.dropout = nn.Dropout(p=0.5)
    return classifier


def validation(model, validation_dataloader, criterion, device_type='cuda'):
    accuracy = 0
    test_loss = 0
    for images, labels in validation_dataloader:
        # Move input and label tensors to the GPU
        images, labels = images.to(device_type), labels.to(device_type)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def trainNetwork(model, training_dataloader, validation_dataloader, lr=0.001, epochs=1, epochs_completed=0,
                 device_type='cuda'):
    """

    :type epochs: int

    """
    print("Starting training @ : {}".format(datetime.now()))
    epochs = epochs  # 1 for testing loading and saving. TODO: increase this after test is complete. 3 is good.
    steps = 0
    running_loss = 0
    print_every = 40
    model.to(device_type)  # Use GPU

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    for e in range(epochs):
        model.train()  # training mode
        for images, labels in training_dataloader:
            steps += 1

            optimizer.zero_grad()
            # Move input and label tensors to the GPU
            images, labels = images.to(device_type), labels.to(device_type)
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()  # evaluation mode
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():  # turn off gradient calculations
                    test_loss, accuracy = validation(model, validation_dataloader, criterion, device_type=device_type)

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Validation Loss: {:.3f}..".format(test_loss / len(validation_dataloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(validation_dataloader))
                      )
                running_loss = 0
    print("Completed training @ : {}".format(datetime.now()))
    return epochs + epochs_completed


def getCategoryNamesDictionary(cat_to_name_file='cat_to_name.json'):
    with open(cat_to_name_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def trainAndCheckpointModel(model, arch, data_dir, save_dir, epochs_completed=0, epochs=6, lr=0.001,
                            device_type='cuda'):
    epochs_completed = trainNetwork(model, createTrainingDataloader(data_dir), createValidationDataloader(data_dir)
                                    , lr=lr
                                    , epochs=epochs
                                    , epochs_completed=epochs_completed
                                    , device_type=device_type)
    # Training is done, Lets save our work
    saveCheckpointToFile(arch, epochs_completed, model, data_dir, save_dir)
    return model


def initializePretrainedModel(arch, hidden_units, output_units):
    if arch == "VGG13":
        model = models.vgg13(pretrained=True)
    elif arch == "VGG16":
        model = models.vgg16(pretrained=True)
    else:
        raise RuntimeError("Unsupported Architecture")
    freezeParameters(model)
    model.classifier = createClassifier(hidden_units, output_units)
    return model


def saveCheckpointToFile(arch, epochs_completed, model, data_dir, save_dir):
    checkpoint_pth = save_dir + '/''checkpoint.pth'
    torch.save(createCheckpointDictionary(arch, epochs_completed, model, data_dir), checkpoint_pth)
    print("Saved checkpoint to {}, after training for a total of {} epochs".format(checkpoint_pth, epochs_completed))


def createCheckpointDictionary(arch, epochs_completed, model, data_dir):
    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx': createTrainingDataset(data_dir).class_to_idx,
                  'epochs_completed': epochs_completed
        , 'arch': arch
        , 'classifier': model.classifier
                  }
    return checkpoint


def retrieveModelFromCheckpoint(checkpoint_pth):
    model = loadModelFromCheckpoint(checkpoint_pth)
    return model

def loadModelFromCheckpoint(checkpoint_pth):
    checkpoint = torch.load(checkpoint_pth)
    # attach state dictionary from the loaded checkpoint to model
    loaded_state_dict = checkpoint['state_dict']
    arch = checkpoint['arch']
    if arch == "VGG16":
        model = models.vgg16(pretrained=True)
    elif arch == "VGG13":
        model = models.vgg13(pretrained=True)
    else:
        raise RuntimeError("Unrecognized architecture model {}".format(arch))
    #
    # Freeze the parameter of the pretrained model
    #
    freezeParameters(model)
    model.classifier = checkpoint['classifier']
    #
    # Attach attributes to model for retrieval later
    #
    model.load_state_dict(loaded_state_dict)
    model.class_to_idx = checkpoint['class_to_idx']
    model.epochs_completed = checkpoint['epochs_completed']
    model.arch = arch

    print("Loaded model was previously trained for {} epochs".format(checkpoint['epochs_completed']))
    return model
def test_trained_network(model,testing_dataloader):
    # Test out your network!
    model.eval()
    model.to('cuda')  # Use GPU
    criterion = nn.NLLLoss()
    accuracy = 0
    test_loss = 0
    torch.no_grad()
    for images, labels in testing_dataloader:
        test_loss_this_batch, accuracy_this_batch = 0, 0
        images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        test_loss_this_batch += criterion(output, labels).item()
        test_loss += test_loss_this_batch
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy_this_batch += equality.type_as(torch.FloatTensor()).mean()
        accuracy += accuracy_this_batch
        print(
            "Test Loss this batch:{:.3f}, Accuracy this batch:{:.3f}".format(test_loss_this_batch, accuracy_this_batch))

    print("=============Average Loss and Accuracy for testing daataset========")
    print("Test Loss: {:.3f}..".format(test_loss / len(testing_dataloader)),
          "Test Accuracy: {:.3f}".format(accuracy / len(testing_dataloader)))

