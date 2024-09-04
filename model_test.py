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
from sklearn.metrics import accuracy_score, classification_report
import copy
from collections import namedtuple
import os
import random
import shutil
import time
from PIL import Image
import json
import torchvision.models as models
import matplotlib.pyplot as plt
# Additional imports for saving results
import csv


#Test the model with different data 
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

class Test_Dataset(Dataset):#for test purpose same dataloader as
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
num_scene_classes = 7  # Number of scene classes
num_weather_classes = 7  # Number of weather classes
num_timeofday_classes = 4  # Number of time of day classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SceneClassificationModel(num_scene_classes, num_weather_classes, num_timeofday_classes)


# Function to save results to a JSON file
def save_results_to_json(results, model_dir, model_name):
    model_name = model_name.replace('.pth', '')
    model_name = model_name + ".json"
    filepath = os.path.join(model_dir, model_name)
    with open(filepath, 'w') as file:
        json.dump(results, file, indent=4)

# Function to save classification reports to a text file
def save_classification_report(report, model_dir, model_name, attribute):
    model_name = model_name.replace('.pth', '')
    model_name = model_name + "_" +attribute + ".txt"
    filepath = os.path.join(model_dir, model_name)
    with open(filepath, 'w') as file:
        file.write(report)

def load_model(model, model_dir, model_path, device):
    model_path = os.path.join(model_dir, model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    model.to(device)
    return model 

def predict_image(model,img_dir, img_path, scene_to_idx, weather_to_idx, timeofday_to_idx, device):
    """
    Make predictions for a single image using the provided model.
    """
    # Define the image transformation (same as during training)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to the size used during training
        transforms.ToTensor(),
    ])

    # Load and preprocess the image
    image_path = os.path.join(img_dir, img_path )
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        scene_output, weather_output, timeofday_output = model(image)

    # Get the predicted classes
    _, scene_pred = torch.max(scene_output, 1)
    _, weather_pred = torch.max(weather_output, 1)
    _, timeofday_pred = torch.max(timeofday_output, 1)

    # Convert predictions to labels
    idx_to_scene = {v: k for k, v in scene_to_idx.items()}
    idx_to_weather = {v: k for k, v in weather_to_idx.items()}
    idx_to_timeofday = {v: k for k, v in timeofday_to_idx.items()}
    predicted_scene = idx_to_scene[scene_pred.item()]
    predicted_weather = idx_to_weather[weather_pred.item()]
    predicted_timeofday = idx_to_timeofday[timeofday_pred.item()]

    return (predicted_scene, predicted_weather, predicted_timeofday)


# Function to wait for a space key press
def wait_for_space_key():
    print("Mismatch found! Press the enter bar to continue...")
    input()  # This will wait for the user to press Enter (or any key in some environments)

# Confusion Matrix and Classification Report for each task
def display_confusion_matrix(y_true, y_pred, labels, title, model_dir, model_name, attribute):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    model_name = model_name.replace('.pth', '')
    model_name = model_name + "_" +attribute + ".png"
    filepath = os.path.join(model_dir, model_name)
    # Save the confusion matrix to an image file
    plt.savefig(filepath, format='png')
    #plt.show()

# example use case 
# model = load_model(model,'first_model.pth',device)
# predicted_scene, predicted_weather, predicted_timeofday= predict_image(model, 'train', '0a0c3694-37864ee7.jpg', scene_to_idx, weather_to_idx, timeofday_to_idx, device)
# print('Predicted Scene:', predicted_scene)
# print('Predicted Weather:', predicted_weather)
# print('Predicted Time of Day:', predicted_timeofday)
'''
------------------------------------------------------
'''
model_dir = 'run_20240902_103417'# Model Directory
model_name = 'model_epoch_5.pth'# Model Name
'''
------------------------------------------------------
'''

model = load_model(model, model_dir, model_name, device)
det_test = open('det_val.json')
det_test_data = json.load(det_test)
val_dir = 'val'

# scene_missmatch = 0
# weather_missmatch = 0
# timeofday_missmatch = 0

# # Initialize lists to store true labels and predictions for confusion matrix
# true_scenes, true_weathers, true_timeofdays = [], [], []
# pred_scenes, pred_weathers, pred_timeofdays = [], [], []

# start_time = time.time()
# total_time = 0

# # Store individual test results for json file
# test_results = []

# for idx, item in enumerate(det_test_data):
#     # if (idx+1) > 2000:
#     #     break
#     # Extract information for each index
#     image_name_test = item['name']
#     scene_test = item['attributes']['scene']
#     weather_test = item['attributes']['weather']
#     timeofday_test = item['attributes']['timeofday']
#     predicted_scene, predicted_weather, predicted_timeofday = predict_image(model, val_dir, image_name_test, scene_to_idx, weather_to_idx, timeofday_to_idx, device)
#     # print(image_name_test)
#     # print('Predicted Scene:', predicted_scene, '---', 'Scene:', scene_test,)
#     # print('Predicted Weather:', predicted_weather, '---', 'Weather:', weather_test,)
#     # print('Predicted Time of Day:', predicted_timeofday, '---', 'Time of Day:', timeofday_test)


#     true_scenes.append(scene_to_idx[scene_test])
#     true_weathers.append(weather_to_idx[weather_test])
#     true_timeofdays.append(timeofday_to_idx[timeofday_test])

#     pred_scenes.append(scene_to_idx[predicted_scene])
#     pred_weathers.append(weather_to_idx[predicted_weather])
#     pred_timeofdays.append(timeofday_to_idx[predicted_timeofday])

#     if scene_test != predicted_scene:
#         scene_missmatch += 1
#         # print(f"Mismatch in image {image_name_test}: -Wrong Scene-")
#         # print(f"Predicted Scene: {predicted_scene},Weather: {predicted_weather},Time of Day: {predicted_timeofday}")
#         # print(f"Labeled   Scene: {scene_test}, Weather: {weather_test}, Time of Day: {timeofday_test}")
#         # wait_for_space_key()
#     if weather_test != predicted_weather:
#         weather_missmatch += 1
#         # print(f"Mismatch in image {image_name_test}: -Wrong Weather-")
#         # print(f"Predicted Scene: {predicted_scene},Weather: {predicted_weather},Time of Day: {predicted_timeofday}")
#         # print(f"Labeled   Scene: {scene_test}, Weather: {weather_test}, Time of Day: {timeofday_test}")
#         #wait_for_space_key()
#     if timeofday_test != predicted_timeofday:
#         timeofday_missmatch += 1
#         # print(f"Mismatch in image {image_name_test}: -Wrong Time of day")
#         # print(f"Predicted Scene: {predicted_scene},Weather: {predicted_weather},Time of Day: {predicted_timeofday}")
#         # print(f"Labeled   Scene: {scene_test}, Weather: {weather_test}, Time of Day: {timeofday_test}")
#         #wait_for_space_key()
#     test_results.append({
#         "image_name": image_name_test,
#         "true_labels": {
#             "scene": scene_test,
#             "weather": weather_test,
#             "timeofday": timeofday_test
#         },
#         "predicted_labels": {
#             "scene": predicted_scene,
#             "weather": predicted_weather,
#             "timeofday": predicted_timeofday
#         },
#         "mismatches": {
#             "scene_mismatch": scene_test != predicted_scene,
#             "weather_mismatch": weather_test != predicted_weather,
#             "timeofday_mismatch": timeofday_test != predicted_timeofday
#         }
#     })

#     if (idx + 1) % 1000 == 0:
#         elapsed_time = time.time() - start_time
#         print(f"Time taken to test {idx + 1} images: {elapsed_time:.2f} seconds")
#         print(f"#Scene Missmatch = {scene_missmatch}, #Weather Missmatch = {weather_missmatch}, #Timeofday Missmatch = {timeofday_missmatch}")
#         total_time += elapsed_time
#         print(f"Total time: = {total_time:.2f} seconds")
#         start_time = time.time()  # Reset the timer

# #final Summary
# print(f"#Scene Missmatch = {scene_missmatch}, #Weather Missmatch = {weather_missmatch}, #Timeofday Missmatch = {timeofday_missmatch}")
# print(f"Total time: = {total_time:.2f} seconds")

# # Calculate and display accuracy for each task
# scene_accuracy = accuracy_score(true_scenes, pred_scenes)
# weather_accuracy = accuracy_score(true_weathers, pred_weathers)
# timeofday_accuracy = accuracy_score(true_timeofdays, pred_timeofdays)
# print(f"Scene Accuracy: {scene_accuracy:.4f}")
# print(f"Weather Accuracy: {weather_accuracy:.4f}")
# print(f"Time of Day Accuracy: {timeofday_accuracy:.4f}")

# scene_labels = list(scene_to_idx.keys())
# weather_labels = list(weather_to_idx.keys())
# timeofday_labels = list(timeofday_to_idx.keys())

# display_confusion_matrix(true_scenes, pred_scenes, scene_labels, "Scene Confusion Matrix", model_dir, model_name,'Scene')
# display_confusion_matrix(true_weathers, pred_weathers, weather_labels, "Weather Confusion Matrix", model_dir, model_name,'Weather')
# display_confusion_matrix(true_timeofdays, pred_timeofdays, timeofday_labels, "Time of Day Confusion Matrix", model_dir, model_name,'Timeofday')

# # Classification Report for each task
# scene_classification_report = classification_report(true_scenes, pred_scenes, target_names=scene_labels)
# print("Scene Classification Report:")
# print(scene_classification_report)

# weather_classification_report = classification_report(true_weathers, pred_weathers, target_names=weather_labels)
# print("Weather Classification Report:")
# print(weather_classification_report)

# timeofday_classification_report = classification_report(true_timeofdays, pred_timeofdays, target_names=timeofday_labels)
# print("Time of Day Classification Report:")
# print(timeofday_classification_report)

# save_results_to_json(test_results, model_dir ,model_name)
# save_classification_report(scene_classification_report, model_dir, model_name, 'Scene')
# save_classification_report(weather_classification_report, model_dir, model_name, 'Weather')
# save_classification_report(timeofday_classification_report,  model_dir, model_name, 'Timeofday')

def test_model(model, val_dir, det_test_data, scene_to_idx, weather_to_idx, timeofday_to_idx, device, model_dir, model_name):
    scene_missmatch = 0
    weather_missmatch = 0
    timeofday_missmatch = 0
    # Initialize lists to store true labels and predictions for confusion matrix
    true_scenes, true_weathers, true_timeofdays = [], [], []
    pred_scenes, pred_weathers, pred_timeofdays = [], [], []

    start_time = time.time()
    total_time = 0
     # Store individual test results for JSON file
    test_results = []

    for idx, item in enumerate(det_test_data):
        # Extract information for each index
        image_name_test = item['name']
        scene_test = item['attributes']['scene']
        weather_test = item['attributes']['weather']
        timeofday_test = item['attributes']['timeofday']

        predicted_scene, predicted_weather, predicted_timeofday = predict_image(
            model, val_dir, image_name_test, scene_to_idx, weather_to_idx, timeofday_to_idx, device
        )
        true_scenes.append(scene_to_idx[scene_test])
        true_weathers.append(weather_to_idx[weather_test])
        true_timeofdays.append(timeofday_to_idx[timeofday_test])

        pred_scenes.append(scene_to_idx[predicted_scene])
        pred_weathers.append(weather_to_idx[predicted_weather])
        pred_timeofdays.append(timeofday_to_idx[predicted_timeofday])

        if scene_test != predicted_scene:
            scene_missmatch += 1
        if weather_test != predicted_weather:
            weather_missmatch += 1
        if timeofday_test != predicted_timeofday:
            timeofday_missmatch += 1

        test_results.append({
            "image_name": image_name_test,
            "true_labels": {
                "scene": scene_test,
                "weather": weather_test,
                "timeofday": timeofday_test
            },
            "predicted_labels": {
                "scene": predicted_scene,
                "weather": predicted_weather,
                "timeofday": predicted_timeofday
            },
            "mismatches": {
                "scene_mismatch": scene_test != predicted_scene,
                "weather_mismatch": weather_test != predicted_weather,
                "timeofday_mismatch": timeofday_test != predicted_timeofday
            }
        })

        if (idx + 1) % 1000 == 0:
            elapsed_time = time.time() - start_time
            print(f"Time taken to test {idx + 1} images: {elapsed_time:.2f} seconds")
            print(f"#Scene Missmatch = {scene_missmatch}, #Weather Missmatch = {weather_missmatch}, #Timeofday Missmatch = {timeofday_missmatch}")
            total_time += elapsed_time
            print(f"Total time: = {total_time:.2f} seconds")
            start_time = time.time()  # Reset the timer
     # Final Summary
    print(f"#Scene Missmatch = {scene_missmatch}, #Weather Missmatch = {weather_missmatch}, #Timeofday Missmatch = {timeofday_missmatch}")
    print(f"Total time: = {total_time:.2f} seconds")
    # Calculate and display accuracy for each task
    scene_accuracy = accuracy_score(true_scenes, pred_scenes)
    weather_accuracy = accuracy_score(true_weathers, pred_weathers)
    timeofday_accuracy = accuracy_score(true_timeofdays, pred_timeofdays)
    print(f"Scene Accuracy: {scene_accuracy:.4f}")
    print(f"Weather Accuracy: {weather_accuracy:.4f}")
    print(f"Time of Day Accuracy: {timeofday_accuracy:.4f}")
    scene_labels = list(scene_to_idx.keys())
    weather_labels = list(weather_to_idx.keys())
    timeofday_labels = list(timeofday_to_idx.keys())

    display_confusion_matrix(true_scenes, pred_scenes, scene_labels, "Scene Confusion Matrix", model_dir, model_name,'Scene')
    display_confusion_matrix(true_weathers, pred_weathers, weather_labels, "Weather Confusion Matrix", model_dir, model_name,'Weather')
    display_confusion_matrix(true_timeofdays, pred_timeofdays, timeofday_labels, "Time of Day Confusion Matrix", model_dir, model_name,'Timeofday')
    # Classification Report for each task
    scene_classification_report = classification_report(true_scenes, pred_scenes, target_names=scene_labels)
    print("Scene Classification Report:")
    print(scene_classification_report)

    weather_classification_report = classification_report(true_weathers, pred_weathers, target_names=weather_labels)
    print("Weather Classification Report:")
    print(weather_classification_report)

    timeofday_classification_report = classification_report(true_timeofdays, pred_timeofdays, target_names=timeofday_labels)
    print("Time of Day Classification Report:")
    print(timeofday_classification_report)

    save_results_to_json(test_results, model_dir, model_name)
    save_classification_report(scene_classification_report, model_dir, model_name, 'Scene')
    save_classification_report(weather_classification_report, model_dir, model_name, 'Weather')
    save_classification_report(timeofday_classification_report,  model_dir, model_name, 'Timeofday')
    return{
        scene_accuracy, weather_accuracy, timeofday_accuracy
    }


scene_accuracy, weather_accuracy, timeofday_accuracy = test_model(model, val_dir, det_test_data, scene_to_idx, weather_to_idx, timeofday_to_idx, device, model_dir, model_name)
print(scene_accuracy, weather_accuracy, timeofday_accuracy)