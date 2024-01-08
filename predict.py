import argparse

from futils import process_image, load_model, predict 


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description = "Use a model to make a prediction")
    add_arg = parser.add_argument
    
    add_arg('input', nargs='?', default='./flowers/test/1/image_06752.jpg', help='Filepath to flower image. Default=./flowers/test/1/image_06743')
    
    add_arg('checkpoint', nargs='?', default='resnet50_model.pth', help='Filepath to model checkpoint. Default=resnet50_model.pth')
    
    add_arg('--top_k', type=int, default=1, help='Number of Top K predicted classes to return. Default=5')
    
    add_arg('--category_names', default='cat_to_name.json', help='Filepath to JASON file mapping categories to real class names. Default=cat_to_name.json')
    
    add_arg('--gpu', action="store_true", help='Boolean flag to use GPU.')

    return parser.parse_args()




def main():
    """Main function"""
    
    args = parse_args()
    
    try:
        image_tensor = process_image(image_path=args.input)
    except:
        print(f"I'm sorry but {args.input} is not a valid image filepath.")
        return
    
    print("Loading model!")
    
    try:
        model = load_model(filepath=args.checkpoint)
    except:
        print(f"I'm sorry but {args.checkpoint} is not a valid checkpoint filepath for this application.")
        return
    
    topk = args.top_k
    
    probabilities, class_names = predict(image_tensor, model, topk, mapping_file=args.category_names, gpu=args.gpu)
    
    if topk == 1:
        print(f"The top class with its predicted probability is:")
    else:
        print(f"The top {topk} classes with their predicted probabilities are:")
        
    width = max([len(name) for name in class_names])
    for class_name, probability in zip(class_names, probabilities):
        print (f"{class_name.title(): <{width}}\t{round(float(probability), 5)} \n")
    
       

if __name__ == '__main__':
    main()

