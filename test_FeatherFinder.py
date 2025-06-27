#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 16:00:31 2025

@author: kennyaskelson
"""

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from PIL import Image
import torch.nn.functional as F



# Define transform, This will modify all the photos to match the ResNet expected dimensions
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset in dir assumes folders that are classes which is NOGO, Great_Horned, and osprey
data_dir = 'feathers_dataset'
dataset = datasets.ImageFolder(data_dir, transform=transform)
# Sets the proportion of training and testing
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
# Randomly split into train and test sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders. 8 images per batch and shuffles training data
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Use a pretrained model. Loads resnet 18
model = models.resnet18(pretrained=True)

### Next bits were super important to get good classification on this super small dataset ###
# Freeze all layers. This means weights will not be updated during training. 
for param in model.parameters():
    param.requires_grad = False

# This then replaces the last layer with our own setup and allows it to be updated with training
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
for param in model.fc.parameters():
    param.requires_grad = True

# Send model to device. On mac so can only use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss and optimizer. This only applies to our last layer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

def evaluate(model, loader):
    # Uses model in evaluation mode
    model.eval()
    # Start counters
    correct = 0
    total = 0
    # No gradients during eval (I guess saves time a mem?)
    with torch.no_grad():
        #loops over inputs and labels of test data
        for inputs, labels in loader:
            # puts inputs and labels to cpu device
            inputs, labels = inputs.to(device), labels.to(device)
            # gets class predictions
            outputs = model(inputs)
            # This looks through output, gets the highest score of a class (sent to empty _) and leaves just the class
            _, predicted = torch.max(outputs, 1)
            # Gives number of images in batch and passes it to a total counter
            total += labels.size(0)
            # This is cool! It compares the predicted and lables which creates a tensor 
            # of boolean true falses. Sum then treats true and false as 1 and 0. 
            # Then the sum is passed as a regular number. 
            correct += (predicted == labels).sum().item()
    return correct / total

# Training loop. Sets epochs and intializes vars we want to track
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Enables training mode and allows for dropout
    model.train()
    for inputs, labels in train_loader:
        #passes inputs and labels to cpu device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clears old gradients.
        optimizer.zero_grad()
        # This conducts a "forward pass" of our inputs through the model
        outputs = model(inputs)
        # Calculates error of true labels and outputs
        loss = criterion(outputs, labels)
        # Now this conducts backpropagation to update weights and biases (I think?)
        loss.backward()
        # Here things are updated from the back propogation
        optimizer.step()
        # tracks running loss
        running_loss += loss.item()
        # This looks through output, gets the highest score of a class (sent to empty _) and leaves just the class
        _, predicted = torch.max(outputs.data, 1)
        # Gives number of images in batch and passes it to a total counter
        total += labels.size(0)
        # Training accuracy!
        # This is cool! It compares the predicted and lables which creates a tensor 
        # of boolean true falses. Sum then treats true and false as 1 and 0. 
        # Then the sum is passed as a regular number. 
        correct += (predicted == labels).sum().item()
        val_accuracy = evaluate(model, test_loader)

    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Loss: {running_loss/len(train_loader):.4f}, "
          f"Train Accuracy: {correct/total:.4f}, "
          f"Validation Accuracy: {val_accuracy:.4f}")

## Now we can test our model on a random image

# Load and transform the image
img = Image.open("000015.jpg").convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device)

# Make a prediction
class_names = dataset.classes  

# Uses model in evaluation mode
model.eval()
# No gradients during eval (I guess saves time a mem?)
with torch.no_grad():
    # raw logits
    output = model(input_tensor)  
    # convert to probabilities
    probabilities = F.softmax(output, dim=1)  
    # highest prob and class
    confidence, predicted = torch.max(probabilities, 1)  

predicted_class = class_names[predicted.item()]
percent_confidence = confidence.item() * 100

print(f"Predicted class: {predicted_class} ({percent_confidence:.2f}% confidence)")