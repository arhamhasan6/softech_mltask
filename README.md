# Permutations Generator

## Description

This Python script generates permutations of a list of numbers provided as input. It utilizes the `argparse` library to allow users to specify the list of numbers through command-line arguments.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/arhamhasan6/softech_mltask

Change your working directory to the project folder:
cd task1

bash
Copy code
python permute.py [list_of_numbers]
[list_of_numbers]: Provide a list of numbers separated by spaces. These numbers will be used to generate permutations.
Example
Suppose you want to generate permutations of the numbers 1, 2, and 3. You can run the following command:

Run the script using Python:
python permute.py <list_of_numbers>
e.g
python permute.py 1 2 3

- **Command-line Interface**: The script is user-friendly and can be run from the command line, allowing users to provide their list of numbers as arguments.

- **Efficient Algorithm**: The code employs a recursive algorithm to generate permutations, ensuring efficiency even for longer lists.
=================================================================================================================================================================
################################################################################################################################################################
 

# Question No.2

 

# STL-10 Image Classification with ResNet-50 and Mixed Precision Training

 

This Python code demonstrates image classification using the STL-10 dataset with the ResNet-50 architecture. It also incorporates mixed precision training for improved training speed and memory efficiency.

 

## Requirements

 

Before running the code, make sure you have the following dependencies installed:

 

- Python 3.x
- PyTorch
- torchvision
- numpy

 

You can install PyTorch and torchvision via pip:



## Usage

- Clone or download this repository to your local machine.

- Open a terminal or command prompt and navigate to the directory containing the code.

- Run the provided ipynb file
 
- The code will start training on the initial layers of prertrained ResNet-50 model on the STL-10 dataset. During training, loss and accuracy for each epoch will displayed in the console.

- After training is complete, the code will evaluate the model's performance on the validation and test sets and display the accuracy.

- Confusion matrix for each class is provided accordingly and prediction on random pictures to check accuracy

## Configuration

You can customize the following configuration parameters in the code:

- `BATCH_SIZE`: The batch size for training and evaluation. You can adjust this value based on your GPU memory.64 recommended
 

- `NUM_CLASSES`: The number of classes in the STL-10 dataset. By default, it's set to 10.

 
- `EPOCHS`: The number of training epochs.15 epochs provided sufficient result which we can increase more until overfitting occurs.

 ## Data Preprocessing
 mean and standard deviation calculation, and resizing images according to resnet input size

## Data Augmentation

Applied different transformations to generalize model 
 

## Dataset Split

The code splits the STL-10 dataset into training, validation, and test sets. By default, it uses 80% of the data for training and 20% for validation. The test set is separate and used for evaluating the model's final performance.
Clone the repository:
   ```bash
   git clone https://github.com/arhamhasan6/softech_mltask
   ```

-Deep Learning CNN
# STL-10 Image Classification with RESNET-50

This repository contains code for training and evaluating a deep neural network, RESNET-50, on the STL-10 image classification dataset. The code is written in PyTorch and is designed to be easy to understand and use.

## About the Code

### Purpose
The purpose of this code is to demonstrate how to train a convolutional neural network (RESNET-50) for image classification tasks using the STL-10 dataset. The code covers data preprocessing, model definition, training, and evaluation.

Multiclass-Classification-on-STL-10-dataset-using-FineTuned-Resnet50-and-SVM-Classifier
a) Features extracted from the last fully-connected layer of pretrained Rsenet50 on ImageNet dataset is used to train a multiclass SVM classifier on STL-10 dataset
b) Fine-tuned the ResNet 50 model for the STL-10 dataset, and evaluated the classification performance on the test set before and after fine-tuning with respect to the Class wise Accuracy.
c) saved the model using pickle file to utilise the model for predicting on images , preprocessing 

############################################################################################################################################################################


## Task 3 

-Yolov8n pretrained model 
-Converting labels from json format to yolo darknet text format
-Inferencing on a video
-Changing hyperparameters using yaml file
# YOLOv8 Object Detection with Ultralytics

This project uses YOLOv8, an object detection model, implemented with the Ultralytics library. It allows you to perform object detection tasks on images and videos using a pre-trained YOLOv8 model.

## Prerequisites

Before getting started, ensure you have the following dependencies installed:

- [Python](https://www.python.org/) (Python 3.10 or later recommended)
- [PyTorch](https://pytorch.org/) (with GPU support)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [ffmpeg](https://www.ffmpeg.org/) (for video processing)

You can install Python packages using pip:

```bash
pip install ultralytics torch torchvision torchaudio opencv-python-headless
```

# YOLOv8 Object Detection with Ultralytics

This project uses YOLOv8, an object detection model, implemented with the Ultralytics library. It allows you to perform object detection tasks on images and videos using a pre-trained YOLOv8n model.

Clone this repository:
git clone https://github.com/yourusername/your-repo.git

Download the YOLOv8 pre-trained weights (if not already included):
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
Navigate to the project directory and start object detection training:
cd /path/to/your/project
yolo task=detect mode=train model=/path/to/yolov8n.pt data=data.yaml epochs=2 imgsz=1024 plots=True batch=32
Once training is complete, you can perform object detection on images or videos:

# Detect objects in a video
yolo task=detect mode=predict model=/path/to/weights/best.pt source=/path/to/video.mp4 show=True

# Detect objects in an image
yolo task=detect mode=predict model=/path/to/weights/best.pt source=/path/to/image.jpg show=True
Configuration
You can customize the training configuration by editing the data.yaml file.
For more options and detailed documentation, refer to the Ultralytics documentation.
Results
After training and prediction, you can find the results and weights in the runs directory. The best model weights will be in runs/detect/trainX/weights/best.pt.





