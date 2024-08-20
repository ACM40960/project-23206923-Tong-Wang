# Emotion Detection using Custom CNN

### Overview

This project focuses on training an emotion detection model using a custom Convolutional Neural Network (CNN) from scratch. The model is trained on the Kaggle FER-2013 dataset, which contains grayscale images of faces categorized into seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

### Dataset

The dataset used is the FER-2013 dataset from Kaggle. It consists of 48x48 pixel grayscale images, where the faces have been automatically aligned. The dataset is divided into training and testing subsets.

### Project Structure

Data Cleaning and Preparation: Scripts to remove non-image files and count the number of images in each emotion category.

Data Analysis: Visualizations showing the distribution of images per emotion and sample images from the dataset.

Model Building: Implementation of a custom CNN model using TensorFlow/Keras.

Model Training and Evaluation: Training the model with the dataset, validating its performance, and testing it on unseen data.

### Model Architecture

#### The custom CNN model consists of:

Multiple convolutional layers with ReLU activation.

Max-pooling layers to reduce spatial dimensions.

Batch normalization and dropout layers for regularization.

Dense layers with a final softmax activation for classification.

### Results

The model is trained to classify emotions with a target accuracy. The performance can be further optimized using different techniques such as data augmentation, transfer learning, etc.

### Authors

Tong Wang
