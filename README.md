# deep_learning
some homework codes in learning IE 534/CS 598 

### hw1.py:
Hand implemented fully connected neural network for MNIST data set. The neural net can have any specified shape, i.e. number of layers.

### hw2_convolution_mnist.py:
Very Fast hand implemented convolution neural network for MNIST data set. There is only one convolution layer but arbitrary number of 
fully connected layers following the convolution layer. The time for one epoch run on CPU with 128 batch size is approximately 5-6 seconds.

### hw3_convolution_cifar10_pytorch.py:
Pytorch implemented convolution neural network for cifar10 data set. Used monte carlo method showing that for deep neural network
![equation](https://latex.codecogs.com/gif.latex?F%28x%2C%5Cmathbb%7BE%7D%28%5Ctheta%29%29%20%3D%20%5Cmathbb%7BE%7D%28F%28x%2C%5Ctheta%29%29).
The result is not mathmatically correct but in practice, the equality holds. 

### hw4_resnet.py:
Pytorch implemented Resnet for cifar100 classification

### hw5_rankent.py:
Pytorch implemented deep ranking net constructed in
[this paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf)

### hw6_Gan.py:
Pytorch implemented General advisorial network.

### hw7_language_model:
Pytorch implemented RNN language model for movie reviews. 
