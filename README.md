# i2dl
Notebook some Machine Learning models for Deep Learning

## Autoencoder for MNIST in Pytorch Lightning
In this notebook you will train an autoencoder for the MNIST dataset which is a dataset of handwritten digits.
I will use the PyTorch Lightning framework which makes everything much more convenient.
One application of autoencoders is unsupervised pretraining with unlabeled data and then finetuning the encoder with labeled data. This can increase our performance if there is only little labeled data but a lot of unlabeled data available.
In this exercise I use the MNIST dataset with 60,000 images of handwritten digits, but I do not have all the labels available.
I will then train our autoencoder to reproduce the unlabeled images.
Then I will transfer the pretrained encoder weights and finetune a classifier on the labeled data for classifying the handwritten digits.
## Facial Keypoint Detection
The exercises of this lecture can be subdivided into mainly two parts. The first part in which we re-invented the wheel and implemented the most important methods on our own and the second part, where we start using existing libraries (that already have implemented all the methods). It's about time to start playing around with more complex network architectures.
We've already entered stage two, but with the introduction of convolution neural networks this week, we are given a very powerful tool that we want to explore in this exercises. Therefore, in this week's exercise your task is to build a convolution neural network to perform facial keypoint detection.
## Semantic Segmentation
In this exercise I am going to work on a computer vision task called semantic segmentation. In comparison to image classification the goal is not to classify an entire image but each of its pixels separately. This implies that the output of the network is not a single scalar but a segmentation with the same shape as the input image. \n
In this Project I implement 2 Approach. First approach Using already implemented models and update the last layer to fit with the project. Second Approach Using transfer learning to get feature extraction from other model and create generice decoder to fit with the project.
## Laboratory 2: Computer Vision MNIST Digit Classification
In the first portion of this lab, we will build and train a convolutional neural network (CNN) for classification of handwritten digits from the famous MNIST dataset. The MNIST dataset consists of 60,000 training images and 10,000 test images. Our classes are the digits 0-9.
### 
## Laboratory 3: Reinforcement Learning (RL)
Reinforcement learning (RL) is a subset of machine learning which poses learning problems as interactions between agents and environments. It often assumes agents have no prior knowledge of a world, so they must learn to navigate environments by optimizing a reward function. Within an environment, an agent can take certain actions and receive feedback, in the form of positive or negative rewards, with respect to their decision. As such, an agent's feedback loop is somewhat akin to the idea of "trial and error", or the manner in which a child might learn to distinguish between "good" and "bad" actions.

In practical terms, our RL agent will interact with the environment by taking an action at each timestep, receiving a corresponding reward, and updating its state according to what it has "learned". 
