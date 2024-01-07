import argparse

from futils import load_data, model_setup, train, test_model_accuracy, save_checkpoint


#python my_python_script.py --input_flag input_value

def parse_args():
    """Parse command line arguments."""
    # Initialization
    args = parse_args()
    
    def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description = "Train a model")
    
    add_arg = parser.add_argument
    add_arg('data_directory', default='./flowers', help='Path to data. Default = ./flowers')
    
    add_arg('--arch', 'Model Name', default='resnet50', help='Choose a model from: resnet50, vgg16, alexnet')
    add_arg('--lr', 'Learning Rate', type=float, default=0.001)
    add_arg('--epochs', 'Num Epochs', type=int, default=5)
    add_arg('--gpu', default='True', help='Choose True or False')
    add_arg('--hidden_units', 'Hidden Units', type=int,)
    
    return parser.parse_args()



def main():
    """Main function"""
    
    args = parse_args()
            
    trainloader, validationloader, testloader, num_classes = load_data(args.data_directory)
            
    # Build the model and optimizer
    model_to_be_trained, criterion, optimizer = model_setup(pretrained_network=args.arch, 
                                                            learing_rate=args.lr, 
                                                            num_classes, 
                                                            hidden_units=args.hidden_units)
    if not model_to_be_trained:
        print(f"I'm sorry but {args.arch} is not a valid model choice for this application.")
        print("Please choose resnet50, vgg16, or alexnet as a pretrained model to fine-tune.")
        return
    
    # Train model and print out training loss, validation loss, and validation accuracy as model trains
    model = train(model_to_be_trained, 
                  trainloader, 
                  validationloader, 
                  criterion, 
                  optimizer,
                  epochs=args.epochs, 
                  args.gpu)
    
    print('Finished training!')
    
    print('Evaluating Model Accuracy on Test Dataset')
    
    test_model_accuracy(model, data_loader=testloader, args.gpu)
    
    print('Saving Model')
    
    save_checkpoint(model, f"{args.arch}_model.pth")
    
    print('All done!')
    

if __name__ == '__main__':
    main()
