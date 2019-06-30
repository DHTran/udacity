import argparse
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import json
import torch
from pathlib import Path
from train import run_validation
from data_utilities import process_image, convert_to_array, resize_and_crop, get_image_for_prediction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_input_args():
    """
    """
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--image_path', type = str, default = 'flowers/test/1/image_06743.jpg', 
                        help = "passes image_path to be process and predict, default = 'flowers/test/1/image_06743.jpg'")
    parser.add_argument('--predict', default = False, action = 'store_true', 
                        help = "runs loaded model to predict image category, default = False")
    parser.add_argument('--checkpoint_path', type = str, default = 'saved_models/checkpoint.pth', 
                        help = "directory to save models, default = 'saved_models/checkpoint.pth'")
    parser.add_argument('--gpu', default = False, action = 'store_true', 
                        help = 'use GPU')
    parser.add_argument('--topk', type = int, default = 1, 
                        help = "prints top prediction (default = 1) or top k predictions")
    parser.add_argument('--json_path', type = str, default = None,
                        help = "enter filepath to json, to load custom category to name mapping")
    in_args = parser.parse_args()
    return parser.parse_args()
                        
def load_model(filepath):
    """loads model from torch.save
    """
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    epochs = checkpoint['epochs']
    learning_rate = checkpoint['learning_rate']
    model.to(device)
    return model, epochs, learning_rate, checkpoint


def show_image_and_top5(image_path, predictions, image_name):
    """plots image sent to prediction, classes, and probabilities
    """
    #convert predictions tuple to list of probabilities and classes
    probs = []
    classes = []
    #convert list of tuples to individual lists of tuple elements
    for item_tuple in predictions:
        probs.append(item_tuple[0])
        classes.append(item_tuple[2]) 
    #print (probs, classes)
    plt.figure(figsize = (6,10))
    ax = plt.subplot (2,1,1)
    title = image_name
    image_np = process_image(image_path, to_tensor = False)
    imshow(image_np, ax, title=title)
    #plot classes and probabilities
    plt.subplot(2,1,2)
    base_color = sb.color_palette()[0]
    sb.barplot(x=probs, y=classes, color = base_color)

def predict(image_tensor, model, class_to_idx, cat_to_name, topk_num = 5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    results = []
    idx_to_class = {}
    for key, value in class_to_idx.items():
        idx_to_class[value] = key    
    with torch.no_grad():
        image = image_tensor.to(device)
        output = model(image)
        ps = torch.exp(output)
        probs, classes = ps.topk(topk_num, dim=1)
        #print (probs, classes)
        for prob, class_ in zip(probs[0], classes[0]):
            index = str(idx_to_class[class_.item()])
            cat_name = cat_to_name[index]
            prob_class = (prob.item(), cat_name)
            #appends tuple of (prob, class)
            results.append(prob_class)
        return results

def main():
    in_args = get_input_args()
    print (f"loading saved model: {in_args.checkpoint_path}")
    model, epochs, learning_rate, checkpoint = load_model(in_args.checkpoint_path)
    image_cat_index = in_args.image_path.split('/')[2]
    print (f"loading image from: {in_args.image_path}")
    image_name = checkpoint['cat_to_name'][image_cat_index]
    print (f"Image name: {image_name}")
    np_image, image_tensor = get_image_for_prediction(in_args.image_path)
    #imshow(np_image)
    if in_args.json_path is not None: 
        with open(cat_to_name, 'r') as f:
            cat_to_name = json_load(f)
    else:
        cat_to_name = checkpoint['cat_to_name']
    if in_args.predict:
        predictions = predict(image_tensor, model, checkpoint['class_to_idx'], cat_to_name, in_args.topk)
        for item in predictions:
            probability = item[0]
            name = item[1]
            print (f"Probability = {probability:.3f}.  Name = {name}")
            
    
if __name__ == '__main__':
    main()