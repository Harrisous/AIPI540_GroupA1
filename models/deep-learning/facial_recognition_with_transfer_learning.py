# -*- coding: utf-8 -*-
"""
### Transfer Learning
"""

from google.colab import drive
drive.mount('/content/drive')

"""
download the going_modular directory from the pytorch-deep-learning repository.
"""

# We need torch 1.12+ and torchvision 0.13+
try:
    import torch
    import torchvision
    assert int(torch.__version__.split(".")[1]) >= 12, "torch version should be 1.12+"
    assert int(torchvision.__version__.split(".")[1]) >= 13, "torchvision version should be 0.13+"
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
except:
    print(f"[INFO] torch/torchvision versions not as required, installing nightly versions.")
    !pip3 install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    import torch
    import torchvision
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")

# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms

# Get torchinfo. Install it if it doesn't work
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")
    !pip install -q torchinfo
    from torchinfo import summary
    
# Try to import the going_modular directory from someone else who have already made the data_setup
try:
    from going_modular.going_modular import data_setup, engine
except:
    # Get the going_modular scripts
    print("[INFO] Couldn't find going_modular scripts... downloading them from GitHub.")
    !git clone https://github.com/mrdbourke/pytorch-deep-learning
    !mv pytorch-deep-learning/going_modular .
    !rm -rf pytorch-deep-learning
    from going_modular.going_modular import data_setup, engine

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

"""Download the dataset from the shared Google Drive Folder and unzip it."""

import os
import zipfile
from pathlib import Path
import requests

# Set path to data folder
data_path = Path("/content/drive/MyDrive/AIPI540A1/data/Split")
image_path = Path(data_path)

# Directories
train_dir = image_path / "train"
val_dir = image_path / "val"
test_dir = image_path / "test"

"""Transformation steps:

1. Batches of size [batch_size, 3, height, width].

2. Values between 0 & 1.

3. Color Jitter: adjust brightness values ranging from 0 - 2, contrast 0 - 1, saturation 0 - 1, and hue +- 0.2 to give a grater range of images to train, reducing overfitting.

4. Normalize: A mean of [0.485, 0.456, 0.406] across each color channel which is the default value of torchvision.transforms.Normalize() function. A standard deviation of [0.229, 0.224, 0.225] across each color channel.


"""

# Create a transforms pipeline manually.
manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=[-0.2,0.2]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

"""Create training and testing DataLoaders."""

# Create training and validation DataLoaders and return a list of class names
train_dataloader, val_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=val_dir,
                                                                               transform=manual_transforms,
                                                                               batch_size=8)

train_dataloader, val_dataloader, class_names

"""Set default weights"""

# Get a set of pretrained model weights
weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
weights

# Get the transforms used to create our pretrained weights
auto_transforms = weights.transforms()
auto_transforms

# Create training and validation DataLoaders and a list of class names
train_dataloader, val_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                          test_dir=val_dir,
                                          transform=auto_transforms, # perform same data transforms on our own data as the pretrained model
                                          batch_size=8) # set mini-batch size to 8

train_dataloader, val_dataloader, class_names

device = "cuda" if torch.cuda.is_available() else "cpu"
device

# Load the model with pretrained weights and send it to the target device (torchvision v0.13+)
weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
model = torchvision.models.efficientnet_b5(weights=weights).to(device)

"""

Parameters:

1. model - the model we work with.

2. input_size - the shape of the data we'd like to pass to our model, which is (batch_size, 3, 224, 224) for efficientnet_b3, the input size is. You can try this out by passing different size input images to summary() or your models.

3. col_names - the various information columns to see about our model.

4. col_width - how wide the columns should be for the summary.

5. row_settings - what features to show in a row.
"""

# Print a summary using torchinfo
summary(model=model,
        input_size=(8, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

"""
Process: 
1. Freeze some base layers of a pretrained model, especially the features selection.

2. Adjust output layers (or the classification layers).

3. Freeze every other layer in the network to retain the weights from other parts of the model.

'requires_grad=False' freezes layers in the feature selection so that PyTorch won't track gradient updates
"""

# Freeze base layers in the "features" section of the model
for param in model.features.parameters():
    param.requires_grad = False

"""
Keep Dropout layer (skip connections betwen 2 NN layers with a probability p) and the same with torch.nn.Dropout(p=0.2, inplace=True).
"""

# set seeds
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# set output shape
output_shape = len(class_names)

# modify the out_features to be the number of classes
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=2048,
                    out_features=output_shape,
                    bias=True)).to(device)

# Summary
summary(model,
        input_size=(8, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

"""
Because we're working with multi-class classification, we'll use nn.CrossEntropyLoss() for the loss function.
Use torch.optim.Adam() as our optimizer with lr=0.001.
"""

# Define loss as cross entropy
loss_fn = nn.CrossEntropyLoss()

# Use Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.008) # originally 0.001

# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Setup training and save the results
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=val_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=10,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# someone created the plotting functions so we will use these to plot the loss and accuracy curves
try:
    from helper_functions import plot_loss_curves
except:
    print("[INFO] Couldn't find helper_functions.py, downloading...")
    with open("helper_functions.py", "wb") as f:
        import requests
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
        f.write(request.content)
    from helper_functions import plot_loss_curves

# Plot the loss curves of our model
plot_loss_curves(results)

import pandas as pd

image_path

Image.open(image_path)

from typing import List, Tuple
from PIL import Image

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module, image_path: str, class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224), transform: torchvision.transforms = None,
                        device: torch.device=device):

    # 1. open image
    img = Image.open(image_path) # /content/drive/MyDrive/AIPI540A1/data/Split/val/Dave

    # 2. create transformation for image
    if transform is None:

        # apply the same transformations as before.
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=[-0.2, 0.2]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ###

    # 3. Make sure the model is on the target device
    model.to(device)

    # 4. Turn on model evaluation mode and inference mode
    model.eval()

    with torch.inference_mode():
      # 5. Transform and add extra dimension to image 
      transformed_image = image_transform(img).unsqueeze(dim=0) # (model requires samples in [batch_size, color_channels, height, width])

      # 6. Make a prediction on image with an extra dimension and send to the target device
      target_image_pred = model(transformed_image.to(device))

    # 7. convert logits to prediction probabilities 
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 8. convert prediction probabilities to prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    prediction = class_names[target_image_pred_label]

    # 9. Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {prediction} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False);

    return prediction, target_image_pred_probs


import pandas as pd

val_image_path_list

names = [path.name for path in val_image_path_list]
names

# Path(ip).glob("*.jpg")

# get list all image paths from test data
val_image_path_list = list(Path(val_dir).glob("*"))

# initialize a list of classification accuracies
data = {}
names = ['Dave', 'Haochen', 'Xiaoquan', 'Harshitha']

for name in names:
    data[name] = {}
    for other_name in names:
        data[name][other_name] = 0

# Create the DataFrame
df = pd.DataFrame(data)
df = df.rename_axis(index=None) # removes default index name
df = df.set_index(df.columns) # sets row index to column names

for each_path in val_image_path_list:
  real = each_path.name
  ip = list(Path(each_path).glob("*.jpg"))
  for i in ip:
    # Make predictions on and plot the images
    pred = pred_and_plot_image(model=model, image_path=i, class_names=class_names, image_size=(224, 224))[0]
    print(pred)
    df.loc[pred, real] += 1 # rows: prediciton, columns: prediction
print(df)

"""
for image_path in val_image_path_list:
    pred_and_plot_image(model=model, image_path=image_path, class_names=class_names, image_size=(224, 224))
"""

val_image_path_list = list(Path(val_dir).glob("*/*.jpg")) # get list all image paths from test data

# Make predictions on and plot the images
for image_path in val_image_path_list:
    pred_and_plot_image(model=model, image_path=image_path, class_names=class_names, image_size=(224, 224))

test_image_path_list = list(Path(test_dir).glob("*/*.jpg")) # get list all image paths from test data
test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                       k=num_images_to_plot) # randomly select 'k' image paths to pred and plot

# Make predictions on and plot the images
for image_path in test_image_path_sample:
    pred_and_plot_image(model=model,
                        image_path=image_path,
                        class_names=class_names,
                        image_size=(224, 224))

