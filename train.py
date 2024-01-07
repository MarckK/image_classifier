import argparse

from futils import load_data, model_setup, train, test_model_accuracy, save_checkpoint



def parse_args():
    parser = argparse.ArgumentParser(description = "Train a model")
    add_arg = parser.add_argument
    
    add_arg('--data_directory', default='./flowers', help='Path to data. Default=./flowers')
    add_arg('--arch', default='resnet50', help='Choose a model from: resnet50, vgg16, alexnet. Default=resnet50')
    add_arg('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
    add_arg('--epochs', type=int, default=5, help='Number of Epochs. Default=5')
    add_arg('--gpu', action="store_true", help='Boolean flag to use GPU.')
    add_arg('--hidden_units', type=int, help='Hidden Units')

    return parser.parse_args()




def main():
    """Main function"""
    
    args = parse_args()
            
    trainloader, validationloader, testloader, num_classes = load_data(args.data_directory)
            
    # Build the model and optimizer
    try:
        model_to_be_trained, criterion, optimizer = model_setup(pretrained_network=args.arch, 
                                                            learing_rate=args.lr, 
                                                            num_classes, 
                                                            hidden_units=args.hidden_units)
    except:
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
                  gpu=args.gpu)
    
    print('Finished training!')
    
    print('Evaluating Model Accuracy on Test Dataset')
    
    test_model_accuracy(model, data_loader=testloader, gpu=args.gpu)
    
    print('Saving Model')
    
    save_checkpoint(model, f"{args.arch}_model.pth")
    
    print('All done!')
    

if __name__ == '__main__':
    main()
