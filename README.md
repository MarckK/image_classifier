# Image Classifier 

In this project, you'll train an image classifier to recognize different species of flowers. 
You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. 
In practice, you'd train this classifier, then export it for use in your application. 
We'll be using this dataset of 102 flower categories.
  
When you've completed this project, you'll have an application that can be trained on any set of labelled images. 
Here your network will be learning about flowers and end up as a command line application. 
But, what you do with your new skills depends on your imagination and effort in building a dataset.
  
*This is the final Project of the Udacity AI with Python Nanodegree*


## Prerequisites

The code for this project is written in Python 3.10, PyTorch 1.12, and torchvision 0.14. 
These are prerequisites for both the .ipynb and the .py files.

## Command Line Application
* Train a new network on a data set with ```train.py```
  * Basic Usage : ```python train.py data_directory```
  * Prints out the current epoch, training loss, validation loss, and validation accuracy as the model trains.
  * Options:
    * Choose model architecture (resnet50, vgg16, alexnet): ```python train.py data_dir --arch "resnet50"```
    * Set hyperparameters: ```python train.py data_dir --lr 0.001 --hidden_units 250 --epochs 5 ```
    * Use GPU for training: ```python train.py data_dir --gpu```
    
* Predict flower name from an image with ```predict.py``` along with the probability of that name. That is, you'll pass in a single image ```/path/to/image``` and return the flower name and class probability.
  * Basic usage: ```python predict.py /path/to/image checkpoint```
  * Options:
    * Return top **K** most likely classes:``` python predict.py input checkpoint ---top_k 5```
    * Select file mapping categories to real names: ```python predict.py input checkpoint --category_names cat_to_name.json```
    * Use GPU for inference: ```python predict.py input checkpoint --gpu```
