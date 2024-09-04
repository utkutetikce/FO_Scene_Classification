#imports
import matplotlib as plt
import numpy as np 
import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import copy
from collections import namedtuple
import os
import random
import shutil
import time
from PIL import Image
import json
import torchvision.models as models 
from torch.utils.tensorboard import SummaryWriter
from model_test import *

det_test = open('det_val.json')
det_test_data = json.load(det_test)
val_dir = 'val'
det_train = 'det_train.json'
train_dir = 'train'

print("code started \n")

scene_to_idx = {
    'tunnel': 0,
    'residential': 1,
    'parking lot': 2,
    'undefined': 3,
    'city street': 4,
    'gas stations': 5,
    'highway': 6,
}
weather_to_idx = {
    'rainy': 0,
    'snowy': 1,
    'clear': 2,
    'overcast': 3,
    'undefined': 4,
    'partly cloudy': 5,
    'foggy': 6,
}
timeofday_to_idx = {
    'daytime': 0,
    'night': 1,
    'dawn/dusk': 2,
    'undefined': 3,
}
print("attribute tables set \n")

class SceneClassificationDataset(Dataset):
    def __init__(self, json_file, img_dir,scene_to_idx, weather_to_idx, timeofday_to_idx, transform=None):
        """
        Args:
            json_file (str): Path to the JSON file with annotations.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        
        self.img_dir = img_dir
        self.transform = transform
        self.scene_to_idx = scene_to_idx  # Assign the scene_to_idx dictionary
        self.weather_to_idx = weather_to_idx  # Assign the weather_to_idx dictionary
        self.timeofday_to_idx = timeofday_to_idx  # Assign the timeofday_to_idx dictionary


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]['name']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)

        # Extract the scene labels
        scene = self.data[idx]['attributes']['scene']
        weather = self.data[idx]['attributes']['weather']
        timeofday = self.data[idx]['attributes']['timeofday']
        scene_label = self.scene_to_idx[scene]  # Convert to numeric label
        weather_label = self.weather_to_idx[weather]  # Convert to numeric label
        timeofday_label = self.timeofday_to_idx[timeofday]  # Convert to numeric label
        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, scene_label, weather_label, timeofday_label
    
#downsampling the images for less computional power
downsample_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Downsample to 128x128
    transforms.ToTensor(), #transforming it to tensor
])

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SceneClassificationModel(nn.Module):
    def __init__(self, num_scene_classes, num_weather_classes, num_timeofday_classes):
        super(SceneClassificationModel, self).__init__()
        
        # Load a pre-trained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Replace the last fully connected layer with three separate linear layers
        # The original ResNet-50 has a fully connected layer with 2048 input features
        self.fc_scene = nn.Linear(2048, num_scene_classes)
        self.fc_weather = nn.Linear(2048, num_weather_classes)
        self.fc_timeofday = nn.Linear(2048, num_timeofday_classes)
    
    def forward(self, x):
        # Pass the input through the ResNet-50 model until the average pooling layer
        x = self.resnet50.avgpool(self.resnet50.layer4(self.resnet50.layer3(self.resnet50.layer2(self.resnet50.layer1(self.resnet50.relu(self.resnet50.bn1(self.resnet50.conv1(x))))))))
        x = torch.flatten(x, 1)
        
        # Get separate outputs for each attribute
        scene_output = self.fc_scene(x)
        weather_output = self.fc_weather(x)
        timeofday_output = self.fc_timeofday(x)
        
        return scene_output, weather_output, timeofday_output
    
# Initialize the model
num_scene_classes = len(scene_to_idx)  # 7 for this case
num_weather_classes = len(weather_to_idx)  # 7 for this case
num_timeofday_classes = len(timeofday_to_idx)  # 4 for this case

model = SceneClassificationModel(num_scene_classes, num_weather_classes, num_timeofday_classes).to(device)
print("model initalized")

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 5

train_dataset = SceneClassificationDataset(det_train, 
                                           train_dir, 
                                           scene_to_idx, 
                                           weather_to_idx, 
                                           timeofday_to_idx, 
                                           transform=downsample_transform)
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

print(f"Loaded training dataset with {len(train_dataset)} samples")
# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # just for test purpose
optimizer = optim.Adam(model.parameters(), lr=learning_rate)    

# Generate a unique directory name based on the current timestamp
run_dir = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(run_dir, exist_ok=True)  # Create directory for the current run
print(f'run director set')

writer = SummaryWriter(run_dir)

# Training loop-----------------------------------------------------------------------------------------------
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0

    for batch_idx, (images, scenes, weather, timeofday) in enumerate(train_loader):
        batch_start_time = time.time()
        # Converting labels to tensor with proper handling
        images = images.to(device)
        scenes = torch.tensor(scenes).clone().detach().long().to(device)
        weather = torch.tensor(weather).clone().detach().long().to(device)
        timeofday = torch.tensor(timeofday).clone().detach().long().to(device)
        # Forward pass
        forward_start_time = time.time()
        scene_outputs, weather_outputs, timeofday_outputs = model(images)
        forward_end_time = time.time()

        # Compute loss
        loss_scene = criterion(scene_outputs, scenes)
        loss_weather = criterion(weather_outputs, weather)
        loss_timeofday = criterion(timeofday_outputs, timeofday)
        loss = loss_scene + loss_weather + loss_timeofday
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        batch_end_time = time.time()
        # Logging times
        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, '
              f'Total Batch Time: {batch_end_time - batch_start_time:.2f} sec, '
              f'Forward Pass Time: {forward_end_time - forward_start_time:.2f} sec')
        
        # Log the loss to TensorBoard
        writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)

    avg_loss = running_loss / len(train_loader)
    epoch_end_time = time.time()

    print(f'Epoch [{epoch+1}/{num_epochs}] completed in {epoch_end_time - epoch_start_time:.2f} sec, '
          f'Average Loss: {avg_loss:.4f}')
     # Log the average loss for the epoch to TensorBoard
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    # Save the model for the current epoch in the run directory
    model_name = f'model_epoch_{epoch+1}.pth'
    model_save_path = os.path.join(run_dir, model_name)
    torch.save(model.state_dict(), model_save_path)
    if (epoch + 1) % 10 == 0:
        model = load_model(model, run_dir, model_name, device)
        scene_accuracy, weather_accuracy, timeofday_accuracy = test_model(model, val_dir, det_test_data, scene_to_idx, weather_to_idx, timeofday_to_idx, device, run_dir, model_name)
        # Log accuracies to TensorBoard
        writer.add_scalar('Accuracy/scene', scene_accuracy, epoch)
        writer.add_scalar('Accuracy/weather', weather_accuracy, epoch)
        writer.add_scalar('Accuracy/timeofday', timeofday_accuracy, epoch)


# Close the TensorBoard writer
writer.close()

#save the model to a file in order to use later
print(f'Model saved to {model_save_path}')
print("Training complete!")   