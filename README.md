# Private_Compress

Demo codes for the AAAI'19 paper *Private Model Compression via Knowledge Distillation*

## Prerequisites

1. Performance test

    - Linux or macOS
    - NVIDIA GPU + CUDA CuDNN 8.0 or CPU(not recommend)
    - Tensorflow-gpu 1.3.0, keras 2.0.5, python 3.6, numpy 1.14.0, scikit-learn 0.18.1

2. Implementation on Android

    - Linux or macOS
    - JDK 1.8
    - Android Studio 2.3.3
    - Android SDK 7.0, Android SDK Build Tools 26.0.1, Android SDK Tools 26.1.1, Android SDK Platform Tools 26.0.1

## Notes

`student_model.py` and `teacher_model.py` are the network classes of student model and teacher model, respectively.

`teacher_convlarge_cifar.npy` stores the weights of the teacher model pretrained on both the public data and the sensitive data of [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

`teacher_convlarge_public.npy` stores the weights of the teacher model pretrained on the public data of [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). It is used to generate adaptive norm bound.

`private-compress-cifar.py` is an example of RONA which trains a compact neural network on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

TFDroid is a demo project on Android system for testing the time overhead of Large-Conv neural network on mobile devices.

## Experimental Setup on CIFAR-10

We detail the experimental setup on CIFAR-10 here. For brevity, we abbreviate the configuration of the neural network as: 

- C = Convolution Layer, with the following number indicating the size of the network, i.e. C5 indicates a convolutional layer with a 5 × 5 kernel 
- S = Stride, i.e. S1 indicates a stride of 1 
- P = Padding, i.e. P0 indicates a padding of zero 
- @ = Number of kernels in Convolution Layer, i.e. @20 indicates 20 kernels in that layer 
- MP = Max Pooling Layer, with the following number indicating the subsampling window, i.e. MP2 indicates max pooling of 2 × 2 windows 
- AP = Average Pooling Layer, with the following number indicating the subsampling process, i.e. AP6-1 indicates average pooling from 6 × 6 to 1 × 1
- FC = Fully Connected Layer, with the following number indicating the number of nodes in that layer

The architecture of the teacher model is: \[C3(S1P0)@128-C3(S1P0)@128-C3(S1P0)@128-MP2(S2)\]-\[C3(S1P0)@256-C3(S1P0)@256-C3(S1P0)@256-MP2(S2)\]-\[C3(S1P0)@512-C3(S1P0)@256-C3(S1P0)@128-MP2(S2)\]-AP6-1-FC10

The architecture of the student model is: \[C3(S1P0)@32-C3(S1P0)@32-C3(S1P0)@32-MP2(S2)\]-\[C3(S1P0)@64-C3(S1P0)@64-C3(S1P0)@64-MP2(S2)\]-\[C3(S1P0)@64-C3(S1P0)@32-C3(S1P0)@32-MP2(S2)\]-AP6-1-FC10

We choose the 4th layer of the teacher model as the hint layer, the 7th layer of the student model as the guided layer. The temperature parameter is set as 3.

The values of other parameters are set as follows: hint_learning_epoch=40, distillation_learning_epoch=8, self_learning_epoch=8, iterations=5, noise_sigma=10, query_select_rate=0.5, self_learning_batchsize=128, hint_distillation_learning_batchsize=512, learning_rate=0.001.  

We preprocessed the data by subtracting per-pixel mean.
