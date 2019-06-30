# Imports here
import torch
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from torch import nn, optim
from pathlib import Path
from torchvision import datasets, transforms, models
from PIL import Image

#uses GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 0.003
epochs = 10

#data directories
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
learning_rate = 0.003
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
train = datasets.ImageFolder(train_dir, train_transforms)
test = datasets.ImageFolder(test_dir, validation_testing_transforms)
validation = datasets.ImageFolder(valid_dir, validation_testing_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = 64)
validation_loader = torch.utils.data.DataLoader(validation, batch_size = 64)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
# TODO: Build and train your network
model = models.vgg16(pretrained = True)
for param in model.parameters():
    param.requires_grad = False

def set_classifier_criterion_optimizer(model, learning_rate):
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
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    return model, criterion, optimizer

model, criterion, optimizer = (
    set_classifier_criterion_optimizer(model, learning_rate))
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
#runs validation on test images
run_validation(model, test_loader)

# TODO: Save the checkpoint 
model.class_to_idx = train.class_to_idx
checkpoint = {
    'input_size' : 25088,
    'output_size' : 102,
    'state_dict' : model.state_dict(),
    'epochs' : epochs, 
    'arch' : 'vgg16',
    'model' : model,
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
    """loads model from torch.save
    """
    checkpoint = torch.load(filepath)
    epochs = checkpoint['epochs']
    learning_rate = checkpoint['learning_rate']
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict']) 
    model.to(device)
    return model, epochs, learning_rate, checkpoint

model, epochs, learning_rate, checkpoint = load_model('checkpoint.pth')

def resize_and_crop(image, short_side = 256):
    """resizes image and maintains aspect ratio
    """
    image = Image.open(image_path)
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

def process_image(image_path, to_tensor = False):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
    resized_img = resize_and_crop(image)
    np_img = convert_to_array(resized_img)
    if to_tensor:
        image_tensor = torch.from_numpy(np_img).type(torch.FloatTensor)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
    else:
        return np_img

def imshow(image, ax=None, title=None):
    """reverses processing of np array to get PIL image
    """
    if ax is None:
        fig, ax = plt.subplots()
    if title: 
        plt.title(title)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1,2,0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

subfolder = '1'
image_path = Path(test_dir+'/'+subfolder+'/'+'image_06743.jpg')
image_tensor = process_image(image_path, to_tensor = True)
image_np = process_image(image_path, to_tensor = False)

def predict(image_path, model, class_to_idx, topk_num = 5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image_tensor = process_image(image_path, to_tensor = True)
    model.eval()
    results = []
    idx_to_class = {}
    for key, value in checkpoint['class_to_idx'].items():
        idx_to_class[value] = key    
    with torch.no_grad():
        image = image_tensor.to(device)
        output = model(image)
        ps = torch.exp(output)
        probs, classes = ps.topk(topk_num, dim=1)
        print (probs, classes)
        for prob, class_ in zip(probs[0], classes[0]):
            index = str(idx_to_class[class_.item()])
            cat_name = cat_to_name[index]
            prob_class = (prob.item(), class_.item(), cat_name)
            #appends tuple of (prob, index, class)
            results.append(prob_class)
        return results


#passes image_path, model, and class_to_idx to predictions, returns list of tuples(prob, idx, class_name)
predictions = predict(
    image_path, model.to(device), checkpoint['class_to_idx'], 5)

# TODO: Display an image along with the top 5 classes
def show_image_and_top5(image_path, predictions, cat_to_name):
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
    title = cat_to_name[subfolder]
    #show image
    np_image = process_image(image_path, to_tensor = False)
    imshow(np_image, ax, title=title)
    #plot classes and probabilities
    plt.subplot(2,1,2)
    base_color = sb.color_palette()[0]
    sb.barplot(x=probs, y=classes, color = base_color)

show_image_and_top5(image_path, predictions, cat_to_name)
