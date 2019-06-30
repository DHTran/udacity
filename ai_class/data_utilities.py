import torch
import json
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

def load_data(data_dir):
    """loads and transforms data using torchvision 
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225]),
    ])
    validation_testing_transforms = transforms.Compose([
    transforms.Resize(250),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225]),
    ])

    train = datasets.ImageFolder(train_dir, train_transforms)
    test = datasets.ImageFolder(test_dir, validation_testing_transforms)
    validation = datasets.ImageFolder(valid_dir, validation_testing_transforms)
    train_loader = torch.utils.data.DataLoader(train, batch_size = 64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test, batch_size = 64)
    validation_loader = torch.utils.data.DataLoader(validation, batch_size = 64)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    class_to_idx = train.class_to_idx
    return class_to_idx, train_loader, test_loader, validation_loader, cat_to_name

def get_image_for_prediction(image_path):
    """opens an image with PIL Image from path
    """
    image = Image.open(image_path)
    resized_img = resize_and_crop(image)
    image_tensor = process_image(resized_img, to_tensor = True)
    np_image = process_image(resized_img, to_tensor = False)
    return np_image, image_tensor
    
def resize_and_crop(image, short_side = 256):
    """resizes image and maintains aspect ratio
    """
    width, height = image.size
    if width < height:
        ratio = height/width
        size = short_side, int(short_side*ratio)
    else:
        ratio = width/height
        size = int(short_side*ratio), short_side
    resized_img = image.resize(size)
    left_margin = (resized_img.width-224)/2
    bottom_margin = (resized_img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    cropped_image = resized_img.crop((left_margin, bottom_margin, 
                                      right_margin, top_margin))
    return cropped_image

def convert_to_array(image):
    """convert image to numpy array: 0-255 channels to 0-1 floats
    """
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2,0,1))
    return np_image

def process_image(image, to_tensor = False):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    np_img = convert_to_array(image)
    if to_tensor:
        image_tensor = torch.from_numpy(np_img).type(torch.FloatTensor)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
    else:
        return np_img