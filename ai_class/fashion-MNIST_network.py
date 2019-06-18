import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Download and load the training data
trainset = datasets.FashionMNIST(
    '~/.pytorch/F_MNIST_data/', 
    download=True, train=True, transform=transform
    )
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)
# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

#previewing an image 
image, label = next(iter(trainloader))
helper.imshow(image[0,:]);

# first section uses part 3 code
# TODO: Define your network architecture here 
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))
# TODO: Create the network, define the criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.003)
# TODO: Train the network here
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # TODO: Training pass
        optimizer.zero_grad()        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()   
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import helper
# Test out your network!
dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)
# TODO: Calculate the class probabilities (softmax) for img
with torch.no_grad():
    logps = model.forward(img)
ps = torch.exp(logps)
# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')

- - -
#from solution for part4
# TODO: Define your network architecture here 
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x
# TODO: Create the network, define the criterion and optimizer
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
# TODO: Train the network here
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        log_ps = model(images)
        loss = criterion(log_ps, labels) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss}")

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import helper
# Test out your network!
dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[1]
# TODO: Calculate the class probabilities (softmax) for img
ps = torch.exp(model(img))
# Plot the image and probabilities
helper.view_classify(img, ps, version='Fashion')