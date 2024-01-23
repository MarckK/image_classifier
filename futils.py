# standard library imports
import os
import json


# third party imports
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image


def check_path(path):
    """ 
    Checks if a given filepath is valid. 
    Returns the filepath if valid, otherwise raises an error.
    """
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")
        

def load_data(filepath='./flowers', gpu=False):
    """
    Applies the transformations (rotations,flips,normalizations and crops) 
    and converts the images to tensor in order to be able to be fed into the neural network
    Args: 
        filepath: Filepath to data [Default: './flowers']
        gpu: Boolean. Indicates the desired device to use for model/network training + eval [Default: False]
    Returns : 
        The dataloaders for the train, validation and test datasets
    """
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

    data_dir = check_path(filepath)
    
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
    class_to_idx = train_dataset.class_to_idx
    
    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size = 32,
                                              shuffle=True,
                                              num_workers=os.cpu_count() if str(device) == "cuda" else 0,
                                              pin_memory=True if str(device) == "cuda" else False)

    validationloader = torch.utils.data.DataLoader(validation_dataset,
                                                   batch_size = 32,
                                                   shuffle=False,
                                                   num_workers=os.cpu_count() if str(device) == "cuda" else 0,
                                                   pin_memory=True if str(device) == "cuda" else False)

    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size = 32,
                                             shuffle=False,
                                             num_workers=os.cpu_count() if str(device) == "cuda" else 0,
                                             pin_memory=True if str(device) == "cuda" else False)
    
    return trainloader, validationloader, testloader, num_classes, class_to_idx


def model_setup(num_classes=102, pretrained_network='resnet50', learning_rate=0.001, hidden_units=None):
    """
    Sets Up a model to train.
    Args: 
        num_classes: The number of classes in training dataset [Default: 102]
        pretrained_network: The pretrained model [Default: resnet50]
        learing_rate: learning rate [Default: 0.001]
        hidden_units: None or Int. Enables user to set units in a hidden layer of the classifier.
    Returns: 
        model: A pretrained model to be further trained or fine-tuned
        criterion: The error function for setting up the optimization problem
        optimizer: The optimization algorithm 
        checkpoint_data: Dictionary. Info on size of layers, to be used when creating model checkpoints
    """

    hidden_layer1 = 1024
    hidden_layer2 = 256
    dropout = 0.2
    
    
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
                               nn.Dropout(dropout),
                               nn.Linear(hidden_layer1, hidden_layer2),
                               nn.ReLU(),
                               nn.Dropout(dropout),
                               nn.Linear(hidden_layer2, num_classes))

    # Swaps out classifier head
    model.fc = classifier
    
    criterion = nn.CrossEntropyLoss()
    # Only trains the classifier parameters, feature parameters remain frozen
    optimizer = optim.Adam(model.fc.parameters(), learning_rate)
    
    model_data = {'arch': pretrained_network,
                  'in_features': in_features,
                  'drop_out': dropout,
                  'hidden_layer1': hidden_layer1,
                  'hidden_layer2': hidden_layer2,
                  'output_layer': num_classes}
    
    return model, criterion, optimizer, model_data


def run_evaluation(model, data_loader, criterion, device):
    """
    Evaluates model for accuracy and loss
    Args:
        model: The model/network to evaluate
        data_loader: The dataloader for evaluation dataset
        criterion: The error function
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
        optimizer: The optimization algorithm 
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

                eval_accuracy, eval_loss = run_evaluation(model, validationloader, criterion, device)

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


def test_model_accuracy(model, data_loader, criterion, gpu=False):
    """
    Evaluates model for accuracy and loss using test dataset.
    Args:
        model: The model/network to evaluate
        data_loader: The dataloader for test dataset
        criterion: The error function
        gpu: Boolean. Indicates the desired device to use for model/network training + eval [Default: False]
    Returns:
        Tuple of test accuracy, test loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    
    test_accuracy, test_loss = run_evaluation(model, data_loader, criterion, device)

    return test_accuracy, test_loss


def save_checkpoint(model, model_data, class_to_idx, filepath='resnet50_model.pth'):
    """
    Saves a model checkpoint to a given filepath location.
    Args:
        model: The model/network to evaluate
        model_data: Dictionary. Info on model, arch, size of layers, etc, to be used when creating model checkpoints
        class_to_idx: Mapping of class number to idx
        filepath: Filepath to where model's checkpoint will be saved [Default='resnet50_model.pth']
    Returns:
        None
    """
    model_data['state_dict'] = model.state_dict()
    model_data['class_to_idx'] = class_to_idx

    torch.save(model_data, filepath)


def load_model(filepath='resnet50_model.pth'):
    """
    Loads a model's checkpoint data and builds a model using that data.
    Args:
        filepath: Filepath to model checkpoint file [Default='resnet50_model.pth']
    Returns:
        The model built with data from the model's checkpoint. 
    """
    filepath = check_path(filepath)
    
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['arch']
    
    if arch == 'resnet50':
        model = models.resnet50(weights = False)
    elif arch == 'vgg16':
        model = models.vgg16(weights = False)
    elif arch == 'alexnet':
        model = models.alexnet(weights = False)
    else:
        return
    
    # Define classifier using checkpoint data
    classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer1']),
                               nn.ReLU(),
                               nn.Dropout(checkpoint['dropout']),
                               nn.Linear(checkpoint['hidden_layer1'], checkpoint['hidden_layer2']),
                               nn.ReLU(),
                               nn.Dropout(checkpoint['dropout']),
                               nn.Linear(checkpoint['hidden_layer2'], checkpoint['output_size']))
    
    
    
    # Swap model's classifier with classifier defined above using checkpoint data.
    model.fc = classifier

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path='./flowers/test/1/image_06743'):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model.
    Args:
        image_path: Filepath to an image to process
    Returns:
        A Tensor representing the processed image.
    """
    filepath = check_path(image_path)
    
    # Load and preprocess the image
    image = Image.open(filepath).convert("RGB")

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
    mapping_file = check_path(mapping_file)
    
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