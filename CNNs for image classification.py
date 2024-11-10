#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm.notebook import tqdm
import sys


# In[9]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # transform images to tensor and normalize

trainval = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform) # load training data

trainset, valset = torch.utils.data.random_split(trainval, [0.9, 0.1]) # split into train and validation sets

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2) # data loader for training set

valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                        shuffle=True, num_workers=2) # data loader for validation set

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform) # load test data

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2) # data loader for test set

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # label names for CIFAR-10 classes


# In[11]:


import matplotlib.pyplot as plt
import numpy as np

# function to display an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get a batch of random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# display images
imshow(torchvision.utils.make_grid(images))
# print class labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[15]:


import torch.nn as nn
import torch.nn.functional as F

# Define a convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer: 3 input channels, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # Output layer for 10 classes

    def forward(self, x):
        # Convolution -> ReLU -> pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten feature maps
        x = x.view(-1, 16 * 5 * 5)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the network
net = Net()
print(net)


# In[17]:


import torch.optim as optim

# Move the network to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[25]:


import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Define device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentation and preprocessing (resize to 224x224 for AlexNet)
transform = transforms.Compose([
    transforms.Resize(224),  # Resize images to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset without re-downloading
trainval = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainset, valset = torch.utils.data.random_split(trainval, [45000, 5000])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

# Data loaders
batch_size = 64
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Function to select a model by name
def get_model(model_name):
    if model_name == "ResNet18":
        model = torchvision.models.resnet18(weights=None)  # No pretrained weights to avoid data leakage
        model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust output layer for CIFAR-10
    elif model_name == "AlexNet":
        model = torchvision.models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
    elif model_name == "GoogleNet":
        model = torchvision.models.googlenet(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model.to(device)

# Training and evaluation function
def train_and_evaluate(model, trainloader, valloader, num_epochs=10, learning_rate=0.0005, model_name="model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_vloss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward, backward, optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss / len(trainloader):.4f}')

        # Validation phase
        model.eval()
        eval_losses = []
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                eval_losses.append(loss.item())

                # Calculate validation accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        eval_loss = np.mean(eval_losses)
        val_accuracy = 100 * correct / total

        print(f'Epoch {epoch+1}, Validation Loss: {eval_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Save best model by validation loss
        if eval_loss < best_vloss:
            best_vloss = eval_loss
            torch.save(model.state_dict(), f'best_{model_name}.pt')
            print(f"Model {model_name} with best validation loss saved.")

# Evaluation function on test set
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Train and test each model
model_names = ["ResNet18", "AlexNet", "GoogleNet"]
for model_name in model_names:
    print(f"\nTraining {model_name}...")
    model = get_model(model_name)
    train_and_evaluate(model, trainloader, valloader, num_epochs=10, learning_rate=0.0005, model_name=model_name)

    # Load best model for testing
    model.load_state_dict(torch.load(f'best_{model_name}.pt'))
    test_accuracy = evaluate(model, testloader)
    print(f'Test Accuracy of {model_name}: {test_accuracy:.2f}%')


# In[ ]:


# Define GoogleNet model
def get_googlenet_model():
    model = torchvision.models.googlenet(weights=None, aux_logits=True)  # Enable auxiliary classifiers
    model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust output layer for CIFAR-10 
    model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, 10)  # Adjust auxiliary classifier 1 output
    model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, 10)  # Adjust auxiliary classifier 2 output
    return model.to(device)

# Train and validate GoogleNet
def train_and_evaluate_googlenet(model, trainloader, valloader, num_epochs=10, learning_rate=0.0005):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_vloss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward, backward, optimize
            outputs = model(inputs)

            # GoogleNet outputs; use main output for loss calculation
            if isinstance(outputs, torchvision.models.GoogLeNetOutputs):
                main_output, aux1_output, aux2_output = outputs
                loss1 = criterion(main_output, labels)
                loss2 = criterion(aux1_output, labels)
                loss3 = criterion(aux2_output, labels)
                loss = loss1 + 0.3 * (loss2 + loss3)  # GoogleNet loss combination
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss / len(trainloader):.4f}')

        # Validation phase
        model.eval()
        eval_losses = []
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Use main output for validation loss calculation
                if isinstance(outputs, torchvision.models.GoogLeNetOutputs):
                    outputs = outputs.logits

                loss = criterion(outputs, labels)
                eval_losses.append(loss.item())

                # Calculate validation accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        eval_loss = np.mean(eval_losses)
        val_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}, Validation Loss: {eval_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Save best model
        if eval_loss < best_vloss:
            best_vloss = eval_loss
            torch.save(model.state_dict(), 'best_GoogleNet.pt')
            print("Model GoogleNet with best validation loss saved.")

# Test GoogleNet
def evaluate_googlenet(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, torchvision.models.GoogLeNetOutputs):
                outputs = outputs.logits  # Use main output for testing

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Start training and testing GoogleNet
model = get_googlenet_model()
train_and_evaluate_googlenet(model, trainloader, valloader, num_epochs=10, learning_rate=0.0005)

# Load best model for testing
model.load_state_dict(torch.load('best_GoogleNet.pt'))
test_accuracy = evaluate_googlenet(model, testloader)
print(f'Test Accuracy of GoogleNet: {test_accuracy:.2f}%')


# In[38]:


import matplotlib.pyplot as plt

epochs = list(range(1, 11))

# ResNet-18 data
train_loss_resnet = [1.3962, 0.8837, 0.7038, 0.5994, 0.5341, 0.4737, 0.4266, 0.3913, 0.3527, 0.3232]
val_loss_resnet = [1.0983, 0.8743, 0.7402, 0.6526, 0.5976, 0.5042, 0.5060, 0.4620, 0.4879, 0.4852]
val_accuracy_resnet = [61.62, 69.36, 74.54, 77.22, 79.68, 83.10, 83.04, 83.88, 83.70, 83.90]

# AlexNet data
train_loss_alexnet = [1.6509, 1.2407, 1.0436, 0.9379, 0.8759, 0.8138, 0.7784, 0.7408, 0.7084, 0.6832]
val_loss_alexnet = [1.3869, 1.0601, 0.9938, 0.8605, 0.8786, 0.7750, 0.7585, 0.7350, 0.6955, 0.6921]
val_accuracy_alexnet = [49.34, 62.62, 65.72, 69.96, 70.26, 73.68, 74.34, 74.54, 76.48, 75.84]

# Plot ResNet-18 loss and accuracy curves
plt.figure(figsize=(14, 6))

# ResNet-18 loss curve
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_resnet, label='ResNet-18 Train Loss', marker='o')
plt.plot(epochs, val_loss_resnet, label='ResNet-18 Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ResNet-18 Loss vs Epoch')
plt.legend()

# ResNet-18 validation accuracy curve
plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracy_resnet, label='ResNet-18 Validation Accuracy', marker='o', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('ResNet-18 Validation Accuracy vs Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# Plot AlexNet loss and accuracy curves
plt.figure(figsize=(14, 6))

# AlexNet loss curve
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_alexnet, label='AlexNet Train Loss', marker='o')
plt.plot(epochs, val_loss_alexnet, label='AlexNet Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('AlexNet Loss vs Epoch')
plt.legend()

# AlexNet validation accuracy curve
plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracy_alexnet, label='AlexNet Validation Accuracy', marker='o', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('AlexNet Validation Accuracy vs Epoch')
plt.legend()

plt.tight_layout()
plt.show()


# In[40]:


import pandas as pd

# Table data
data = {
    "Model": ["ResNet-18", "AlexNet"],
    "Best Validation Loss": [0.4852, 0.6921],
    "Best Validation Accuracy (%)": [83.90, 75.84],
    "Test Accuracy (%)": [84.35, 76.09],
    "Learning Rate": [0.0005, 0.001],
    "Batch Size": [32, 64]
}

# Create DataFrame
df = pd.DataFrame(data)
print(df)

# Display table with caption
df.style.set_caption("Performance Comparison on Validation and Test Sets")


# In[ ]:




