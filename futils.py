# standard library imports
import os
from pathlib import Path
import json


# third party imports
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_data(location='./flowers'):
    """
    Applies the transformations (rotations,flips,normalizations and crops) 
    and converts the images to tensor in order to be able to be fed into the neural network

    Args: 
        location: Path to data [Default: './flowers']

    Returns : 
        The dataloaders for the train, validation and test datasets
    """
    
    data_dir = location
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define the transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder, applying transforms
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    num_classes = len(train_dataset)
    
    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size = BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=os.cpu_count() if str(device) == "cuda" else 0,
                                              pin_memory=True if str(device) == "cuda" else False)

    validationloader = torch.utils.data.DataLoader(validation_dataset,
                                                   batch_size = BATCH_SIZE,
                                                   shuffle=False,
                                                   num_workers=os.cpu_count() if str(device) == "cuda" else 0,
                                                   pin_memory=True if str(device) == "cuda" else False)

    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size = BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=os.cpu_count() if str(device) == "cuda" else 0,
                                             pin_memory=True if str(device) == "cuda" else False)

    return trainloader, validationloader, testloader, num_classes


def model_setup(pretrained_network='resnet50', learing_rate=0.001, num_classes=102, hidden_units=None):
    """
    Sets Up a model to train.
    Args: 
        pretrained_network: The pretrained model [Default: resnet50]
        learing_rate: learning rate [Default: 0.001]
        num_classes: The number of classes in training dataset [Default: 102]
        
    Returns: The set up model, along with the criterion and the optimizer for training
    """

    hidden_layer1 = 1024
    hidden_layer2 = 256
    
    
    if pretrained_network == 'resnet50':
        model = models.resnet50(pretrained=True)
        in_features = 2048
        
    elif pretrained_network == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = 4096
        
    elif pretrained_network == 'alexnet':
        model = models.alexnet(pretrained = True)
        in_features=4096
        
    else:
        return

    if hidden_units:
        if hidden_layer2 < hidden_units < in_features:
            hidden_layer1 = hidden_units
        elif num_classes < hidden_units < hidden_layer2:
            hidden_layer2 = hidden_units
    
    # Turns off gradients for the model - freeze the complete network
    for param in model.parameters():
        param.requires_grad = False

    # Defines new classifier
    classifier = nn.Sequential(nn.Linear(in_features, hidden_layer1),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(hidden_layer1, hidden_layer2),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(hidden_layer2, num_classes))

    # Swaps out classifier head
    model.fc = classifier
    
    criterion = nn.CrossEntropyLoss()
    # Only trains the classifier parameters, feature parameters remain frozen
    optimizer = optim.Adam(model.fc.parameters(), lr = learing_rate)
    
    return model, criterion, optimizer


def run_evaluation(model, data_loader, criterion, device=device):
    """
    Evaluates model for accuracy and loss
    Args:
        model: The model/network to evaluate
        data_loader: The dataloader for evaluation dataset
        criterion: The error function for setting up the optimization problem
        device: The device being used for model/network training + eval

    Returns:
        Tuple of evaluation accuracy, evaluation loss
    """
    # Ensure model on default device
    model.to(device)
    # Set model to evaluation mode
    model.eval()

    total_eval_loss = 0
    total_eval_accuracy = 0


    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device).type(torch.long) # typecasting labels as using CE Loss
        with torch.no_grad():
          
          logits = model(images)
          
          batch_loss = criterion(logits, labels)
          
          total_eval_loss += batch_loss.item()
          
          ps = torch.exp(logits)
          
          top_ps, top_class = ps.topk(1, dim=1)
          
          equality = top_class == labels.view(*top_class.shape)
          
          total_eval_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

        eval_loss = total_eval_loss/len(data_loader)
        eval_accuracy = total_eval_accuracy /len(data_loader)

    return (eval_accuracy, eval_loss)



def train(model, trainloader, validationloader, criterion, optimizer, epochs=5, gpu=False):
    """
    Defines the model training pipeline,
    with evaluation of training loss, validation loss, and validation accuracy
    at regular intervals during training; these are printed out as well as
    stored in their respective lists and returned as part of a training statistics dictionary.

    Args:
        model: The model/network to train
        trainloader: The dataloader for training dataset
        validationloader: The dataloader for validation dataset
        criterion: The error function for setting up the optimization problem
        optimizer: The optimization algorithm to use
        epochs: The number of epochs to train the model for [Default: 5]
        gpu: Boolean. Indicates the desired device to use for model/network training + eval [Default: False]

    Output:
        Prints the training loss, validation loss, and validation accuracy, as
        calculated at regular intervals during training.

    Returns:
        A dictionary of training statistics containing training losses (list),
        evaluation losses (list), and validation accuracy (list).
    """
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

    # clearing cache if using GPU
    if str(device) == 'cuda':
        torch.cuda.empty_cache()

    model.to(device)
    # Ensure model on default device
    model.to(device)
    # Set model to train mode
    model.train()

    training_loss = 0
    print_every = 100

    accuracy, train_losses, eval_losses = [], [], []

    # Training loop
    for epoch in range(epochs):
        for step, (images, labels) in enumerate(trainloader, 1):
            images, labels = images.to(device), labels.to(device).type(torch.long)      # typecasting labels as using CE Loss

            logits = model(images)
            loss = criterion(logits, labels)
            training_loss += loss.item()

            # Before calculating the gradients, we need to ensure that they are all zero.
            # Otherwise, the gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()
            # Update the parameters
            optimizer.step()

            # Drop out of the training loop and test the network's accuracy and loss on validation dataset
            if step % print_every == 0:
                train_losses.append(training_loss/len(trainloader))

                eval_accuracy, eval_loss = run_evaluation(model, validationloader)

                eval_losses.append(eval_loss)
                accuracy.append(eval_accuracy)

                print(f"Epoch {epoch+1}    "
                      f"Train loss: {training_loss/step:.3f}    "
                      f"Validation loss: {eval_loss:.3f}    "
                      f"Validation accuracy: {eval_accuracy*100:.2f}%" )

                training_loss = 0
                # Set model back to train mode
                model.train()
                

    training_statistics = {"validation_accuracy": accuracy,
                           "validation_losses": eval_losses,
                           "train_losses": train_losses}

    return training_statistics


def test_model_accuracy(model, data_loader=testloader, gpu=False):
    """
    Evaluates model for accuracy and loss using testdataset.
    Prints the test accuracy.
    Args:
        model: The model/network to evaluate
        data_loader: The dataloader for test dataset [Default: testloader]
        gpu: Boolean. Indicates the desired device to use for model/network training + eval [Default: False]
    Output:
        Prints the test accuracy as a percentage.
    Returns:
        Tuple of test accuracy, test loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    
    test_accuracy, test_loss = run_evaluation(model, data_loader, device)

    print(f"Test Accuracy: {test_accuracy*100:.2f}%")

    return test_accuracy, test_loss


def save_checkpoint(model, filepath='checkpoint.pth', train_dataset=train_dataset):
    """
    Saves the a checkpoint for a model to a given file path.
    """
    checkpoint = {'input_size': 2048,
                'output_size': len(train_dataset.classes),
                'hidden_layer1': HIDDEN_LAYER_1,
                'hidden_layer2': HIDDEN_LAYER_2,
                'dropout': 0.2,
                'state_dict': model.state_dict(),
                'class_to_idx': train_dataset.class_to_idx}

    torch.save(checkpoint, filepath)


def load_model(filepath='checkpoint.pth'):
    checkpoint = torch.load(filepath)
    model = models.resnet50(weights = False)

    # Defines a new classifier
    classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer1']),
                               nn.ReLU(),
                               nn.Dropout(checkpoint['dropout']),
                               nn.Linear(checkpoint['hidden_layer1'], checkpoint['hidden_layer2']),
                               nn.ReLU(),
                               nn.Dropout(checkpoint['dropout']),
                               nn.Linear(checkpoint['hidden_layer2'], checkpoint['output_size']))
    model.fc = classifier

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path='./flowers/test/1/image_06743'):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model.

    Args:
        image_path: Path to an image to process
    Returns:
        A Tensor representing the processed image.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")

    img_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    image_tensor = img_transform(image)

    return image_tensor

def predict(image_tensor, model, topk=1, mapping_file='cat_to_name.json', gpu=False):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    Args:
        image_tensor: the preprocessed image tensor to be used for model inference
        model: model loaded from the model checkpoint file
        topk: Integer denoting number of top probabilites to select [Default: 1]
        mapping_file: JSON file mapping  category numbers to category class names
        gpu: Boolean. Indicates the desired device to use for model/network training + eval [Default: False]
    Returns:
        A tuple containing the probabilities (NumPy array) and classes (NumPy array)
    """
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    
    # Ensure processed image on device
    input_tensor = processed_image_tensor.unsqueeze(0).to(device)

    # Move model to device and switch to eval mode
    model.to(device)
    model.eval()

    # Make predictions
    with torch.no_grad():
        output = model(input_tensor)

    # Convert the output to probabilities using softmax
    probabilities = F.softmax(output[0], dim=0)

    probs, indices = probabilities.topk(topk)
    # Convert probs, indices tensors to NumPy arrays
    probs = probs.numpy(force=True)
    indices = indices.numpy(force=True)

    # invert class_to_idx
    class_to_idx = model.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Get classes (represented by nums, which can be used to map to class flower names)
    classes = [idx_to_class[index] for index in indices]
    
    with open(mapping_file, 'r') as f:
        classnum_to_classname = json.load(f)
        
    class_names = [classnum_to_classname[class_num] for class_num in classes]

    return probs, class_names 


    