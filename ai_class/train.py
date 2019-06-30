import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from torch import nn, optim
from pathlib import Path
from torchvision import datasets, transforms, models
from data_utilities import load_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_input_args():
    """get command line arguments
    --arch = model to train
    --data_dir = dir with image data (default '/flowers/')
    --save_dir = dir to save models (default '/saved_models/')
    --learning_rate = sets learning rate (default = 0.003)
    --epochs = sets number of epochs (default = 1)
    --layers = sets number of hidden layers
    --classes = number of categories of images
    --gpu = flag to use GPU (default = True)
    
    """
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--arch', type = str, default = 'vgg', choices = ['vgg', 'densenet', 'alexnet'],
        help = "sets model to use among these three: 'vgg', 'densenet', 'alexnet'")
    parser.add_argument('--data_dir', type = str, default = 'flowers/', 
        help = "directory with images to train, default = 'flower/'")
    parser.add_argument('--save_dir', type = str, default = 'saved_models/', 
        help = "directory to save models, default = 'saved_models/'")
    parser.add_argument('--learning_rate', type = float, default = 0.003, 
        help = "Learning rate, default = 0.003")
    parser.add_argument('--h_outputs', type = str, default = "1024,512", 
        help = """layer outputs, default = '1024,512'.  Default creates  
        the following classifier layers: (inputs, 1024) -> (1024, 512)
        -> (512, categories), with dropout (p=0.2) and ReLU activation 
        functions""")
    parser.add_argument('--epochs', type = int, default = 1, 
        help = "Number of epochs, default = 1")
    parser.add_argument('--categories', type = int, default = 102, 
        help = "Number of image categories to classify")
    parser.add_argument('--gpu', default = False, action = 'store_true', 
         help = 'use GPU')
    parser.add_argument('--save', default = False, action = 'store_true', 
         help = 'saves model to save directory')
    parser.add_argument('--train', default = False, action = 'store_true', 
                        help = 'train model on data')
    in_args = parser.parse_args()
    return parser.parse_args()


def load_model(arch, h_outputs, categories, learning_rate):
    """load one of three models and returns model and inputs of model
    """
    if arch == 'vgg':
        model = models.vgg16(pretrained = True)
        inputs = 25088
    elif arch == 'densenet':
        model = models.densenet201(pretrained = True)
        inputs = 1920
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
        inputs = 9216
    for param in model.parameters():
        param.requires_grad = False
    classifier = build_classifier(inputs, h_outputs, categories)
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    return model, criterion, optimizer

def build_classifier(inputs, h_outputs, categories):
    """uses modules list to create layers, layer inputs and outputs
    from list of hidden layer units
    """
    modules = []
    in_ = inputs
    for index, item in enumerate(h_outputs):
        layer = []
        layer.append(nn.Linear(in_, item))
        layer.append(nn.ReLU())
        layer.append(nn.Dropout(p=0.2))
        modules.extend(layer)
        in_ = item
    final_layer = []
    final_layer.append(nn.Linear(in_, categories))
    final_layer.append(nn.LogSoftmax(dim=1))
    modules.extend(final_layer)
    #print(modules)
    classifier = nn.Sequential(*modules)
    return classifier

def train_model(model, criterion, optimizer, epochs, 
                train_loader, validation_loader):
    #trains model classifier
    steps = 0 
    running_loss = 0
    print_every = 10
    model.train()
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

def run_validation(model, criterion, loader):
    """calculates accuracy of model classification on test/validation set
    """
    print ("Running Validation")
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
    print(f"Test accuracy on test set: {accuracy/len(loader):.3f}")

def save_model(model, cat_to_name, class_to_idx, epochs, 
               arch, learning_rate, optimizer, criterion, save_dir):
    """method to save model with torch.save
    """
    model.class_to_idx = class_to_idx
    checkpoint = {
        'state_dict' : model.state_dict(),
        'epochs' : epochs, 
        'arch' : arch,
        'model' : model,
        'learning_rate' : learning_rate,
        'batch_size' : 64,
        'optimizer' : optimizer.state_dict(),
        'class_to_idx' : model.class_to_idx,
        'cat_to_name' : cat_to_name,
        'criterion' : criterion
    }
    path = Path(save_dir+'checkpoint.pth')
    torch.save(checkpoint, path)
    
def main():
    in_args = get_input_args()
    h_outputs = [int(x) for x in in_args.h_outputs.split(',')]
    class_to_idx, train_loader, test_loader, validation_loader, cat_to_name = (
        load_data(in_args.data_dir))
    model, criterion, optimizer = load_model(in_args.arch, h_outputs, 
                                             in_args.categories, in_args.learning_rate)
    print (model)
    if in_args.gpu: 
        model.to(device)
    if in_args.train:
        train_model(model, criterion, optimizer, in_args.epochs,
                    train_loader, validation_loader)
        run_validation(model, criterion, test_loader)
    if in_args.save:
        save_model(model, cat_to_name, class_to_idx, in_args.epochs, 
                   in_args.arch, in_args.learning_rate, optimizer, criterion, in_args.save_dir)
    
if __name__ == '__main__':
    main()