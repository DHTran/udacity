# Imports here
import torch
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
#uses GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#data directories
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
learning_rate = 0.003
epochs = 10
# TODO: Define your transforms for the training, validation, and testing sets
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

# TODO: Load the datasets with ImageFolder
train = datasets.ImageFolder(
    train_dir, train_transforms)
test = datasets.ImageFolder(
    test_dir, validation_testing_transforms)
validation = datasets.ImageFolder(
    valid_dir, validation_testing_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(
    train, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(
    test, batch_size = 64)
validation_loader = torch.utils.data.DataLoader(
    validation, batch_size = 64)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
# TODO: Build and train your network
model = models.vgg16(pretrained = True)
for param in model.parameters():
    param.requires_grad = False

def set_classifier_criterion_optimizer(model):
    """function to set classifier network, loss function, optimizer
    """
    classifier = nn.Sequential(nn.Linear(25088, 1024),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(1024 ,512),
                                     nn.ReLU(),
                                     nn.Linear(512, 102),
                                     nn.LogSoftmax(dim = 1),
                                    )
    model.classifier = classifier    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)
    return model.classifier, criterion, optimizer

model.classifer, criterion, optimizer = set_classifier_criterion_optimizer(model)
model.to(device)

#trains model classifier
steps = 0 
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for images, labels in train_loader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim = 1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(validation_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validation_loader):.3f}")
            running_loss = 0
            model.train()

# TODO: Do validation on the test set
#model = 
def run_validation(model, loader):
    """calculates accuracy of model classification on test/validation set
    """
    model.eval()
    test_loss = 0
    accuracy = 0
    batch_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print(f"Test accuracy: {accuracy/len(loader):.3f}")

run_validation(model, test_loader)

# TODO: Save the checkpoint 
model.class_to_idx = train.class_to_idx
checkpoint = {
    'input_size' : 25088,
    'output_size' : 102,
    'state_dict' : model.state_dict(),
    'epochs' : epochs, 
    'arch' : 'vgg16',
    'classifier' : model.classifier,
    'learning_rate' : .003,
    'batch_size' : 64,
    'optimizer' : optimizer.state_dict(),
    'class_to_idx' : model.class_to_idx,
}
print (model)
print (model.state_dict().keys())
torch.save(checkpoint, 'checkpoint.pth')

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(filepath):
    """loads model from saved torch.save
    """
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
    else: 
        print("Sorry I'm just using vgg16 for this project at the moment")
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifer, criterion, optimizer = set_classifier_criterion_optimizer(model)
    model.load_state_dict(checkpoint['state_dict']) 
    epochs = checkpoint['epochs']
    learning_rate = checkpoint['learning_rate']
    model.to(device)
    return model, criterion, optimizer, epochs, learning_rate

    model, criterion, optimizer, epochs, learning_rate = load_model('checkpoint.pth')

def resize_and_crop(image, short_side = 256):
    """resizes image (maintains aspect ratio) and center crops
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
    """convert image to numpy array and normalizes
    """
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    #moves 3rd channel to 1st dimension
    np_image = np_image.transpose((2,0,1))
    return np_image

def imshow(image, ax=None, title=None):
    """reverses transform to check image
    """
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

def predict(image, model, topk_num = 5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        ps = torch.exp(output)
        probs, classes = ps.topk(topk_num, dim=1)
        return probs, classes


test_image = Image.open(valid_dir+'/1/image_06739.jpg')
img_tensor = process_image(test_image)
probs, classes = predict(img_tensor, model.to(device))