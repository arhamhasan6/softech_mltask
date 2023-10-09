# Question No.1
# Permutation Finder

 

This Python script finds all permutations of a given array of numbers using backtracking.

 

## Usage

 

Enter a list of numbers separated by commas when prompted.

 

## Example

 

If you input the following list: [1, 2, 3], the script will find all permutations:

 

Enter the elements of the array: 1, 2, 3
The total permutations are: [[1, 2, 3], [2, 1, 3], [3, 2, 1], [2, 3, 1], [1, 3, 2], [3, 1, 2]]

 

## How It Works

 

The script uses a backtracking algorithm to generate all permutations of the input array.
=================================================================================================================================================================

 

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

 

1. Clone or download this repository to your local machine.

 

2. Open a terminal or command prompt and navigate to the directory containing the code.

 

3. Run the code by executing the following command:

 

4. The code will start training the ResNet-50 model on the STL-10 dataset. During training, you will see epoch-wise loss values displayed in the console.

 

5. After training is complete, the code will evaluate the model's performance on the validation and test sets and display the accuracy.

 

## Configuration

 

You can customize the following configuration parameters in the code:

 

- `BATCH_SIZE`: The batch size for training and evaluation. You can adjust this value based on your GPU memory.

 

- `GRADIENT_ACCUMULATION_STEPS`: The number of gradient accumulation steps. Adjust this value for gradient accumulation to save GPU memory.

 

- `NUM_CLASSES`: The number of classes in the STL-10 dataset. By default, it's set to 10.

 

- `EPOCHS`: The number of training epochs.

 

## Mixed Precision Training

 

This code utilizes mixed precision training for faster training and reduced GPU memory consumption. It leverages the `autocast` and `GradScaler` from PyTorch's `torch.cuda.amp` module.

 

## Dataset Split

 

The code splits the STL-10 dataset into training, validation, and test sets. By default, it uses 80% of the data for training and 20% for validation. The test set is separate and used for evaluating the model's final performance.
1. Clone the repository:

   ```bash
   git clone https://github.com/arhamhasan6/softech_mltask
   ```

2. Run the script:

   ```bash
   python permutation.py 1 2 3
   ```

   You'll be prompted to input a list of unique integers separated by spaces. After providing the input, the script will calculate and show all permutations. You have to enter number after

3. Customize the code or incorporate it into your projects as needed.

## Requirements
- Python 3.11.5

## Note

- To obtain meaningful permutations, make sure to input a list of unique integers separated by spaces.
- This code concentrates on efficiently generating permutations and can serve as a valuable resource for understanding backtracking algorithms.

############################################################################################################################################################################

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
## Usage



2. Install the required libraries :

   ```bash
   pip install torch torchvision numpy
   ```

3. Run the code:

   ```bash
   python train_and_evaluate.py
   ```

   This will train the RESNET-50 model on the STL-10 dataset and print the training loss and test accuracy.


############################################################################################################################################################################


## Task 3 

-Yolov8n pretrained model 
-Converting labels from json format to yolo darknet text format
-Inferencing on a video
-Changing hyperparameters using yaml file


