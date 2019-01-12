import argparse
import numpy as np
import torch
from torch import nn
from torchvision import models
from PIL import Image
from model import create_model
import json

def main():
    # Creates & retrieves Command Line Arguments
    parser = argparse.ArgumentParser(description='Train neural network - Image Classifier')
    parser.add_argument('input_image', action='store', type=str)
    parser.add_argument('checkpoint', action='store', type=str)
    parser.add_argument('--top_k', action='store', dest='top_k', type=int, default=5)
    parser.add_argument('--category_names ', action='store', dest='category_names', type=str)
    parser.add_argument('--gpu', dest='device', action='store_const', const='cuda', default='cpu')
    args = parser.parse_args()
    checkpoint = args.checkpoint
    input_image = args.input_image
    category_names = args.category_names
    top_k = args.top_k
    device = args.device
    
    # Loads model from checkpoint
    model = load_checkpoint(checkpoint)
    
    # Sets criterion
    criterion = nn.NLLLoss()

    # Predicts output & prints results
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        probs, classes = predict(input_image, model, top_k, device, cat_to_name)
    else:
        probs, classes = predict(input_image, model, top_k, device)
    print(probs)
    print(classes)

def load_checkpoint(filepath):
    ''' Loads checkpoint from file
        returns model
    '''
    checkpoint = torch.load(filepath)
    model = create_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    new_size = np.array(image.size) / min(image.size) * 256
    image = image.resize(new_size.astype(int))
    center_x = image.size[0] / 2
    center_y = image.size[1] / 2
    crop_box = (center_x - 112, center_y - 112, center_x + 112, center_y + 112)
    image = image.crop(crop_box)
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def predict(image_path, model, topk=5, device='cpu', cat_to_name=None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    img = process_image(Image.open(image_path))
    img = torch.from_numpy(img).float().unsqueeze_(0).to(device)
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img)
    ps = torch.exp(output)
    ps.topk(topk)
    probs, indexes = ps.topk(topk)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    index_list = indexes.tolist()[0]
    classes = []
    for i in range(len(index_list)):
        if cat_to_name:
            classes.append(cat_to_name[str(idx_to_class[index_list[i]])])
        else:
            classes.append(idx_to_class[index_list[i]])
    return probs.tolist()[0], classes

# Call to main function to run the program
if __name__ == "__main__":
    main()