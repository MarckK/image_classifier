import argparse

from futils import load_data, model_setup, train, test_model_accuracy, save_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description = "Train a model")
    add_arg = parser.add_argument
    
    add_arg('data_directory', nargs='?', default='./flowers', help='Path to data. Default=./flowers')
    add_arg('--arch', default='resnet50', help='Choose a model from: resnet50, vgg16, alexnet. Default=resnet50')
    add_arg('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
    add_arg('--epochs', type=int, default=5, help='Number of Epochs. Default=5')
    add_arg('--gpu', action="store_true", help='Boolean flag to use GPU.')
    add_arg('--hidden_units', type=int, help='Hidden Units')

    return parser.parse_args()



def main():
    """Main function"""
    
    args = parse_args()
    
    trainloader, validationloader, testloader, num_classes, class_to_idx = load_data(args.data_directory, gpu=args.gpu)
    
    # Build the model and optimizer
    try:
        model_to_be_trained, criterion, optimizer, checkpoint_data = model_setup(num_classes, 
                                                                     pretrained_network=args.arch, 
                                                                     learning_rate=args.lr, 
                                                                     hidden_units=args.hidden_units)
    except:
        print(f"I'm sorry but {args.arch} is not a valid model choice for this application.")
        print("Please choose resnet50, vgg16, or alexnet as a pretrained model to fine-tune.")
        return
    
    print(f"Fine-tuning pretrained {args.arch} model on data from {args.data_directory} directory.")
    
    # Train model and print out training loss, validation loss, and validation accuracy as model trains
    model = train(model_to_be_trained, 
                  trainloader, 
                  validationloader, 
                  criterion, 
                  optimizer,
                  epochs=args.epochs, 
                  gpu=args.gpu)
    
    print("Finished training!")
    
    print("Evaluating Model Accuracy on Test Dataset")
    
    test_accuracy, test_loss = test_model_accuracy(model, testloader, criterion, gpu=args.gpu)
    
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    print("Saving Model")
    
    save_checkpoint(model, class_to_idx, checkpoint_data, f"{args.arch}_model.pth")
    
    print(f"The model checkpoint is saved to {args.arch}_model.pth")
    print("All done!")
    

if __name__ == '__main__':
    main()
