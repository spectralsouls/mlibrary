Implementation of the neural network described in the following paper: **https://sing.stanford.edu/curis-fellowships/rh/vision-dnn.pdf **

Note: the authors originally implemented AlexNet in 2 GPUS because the model couldn't fit in a single GPU at the time (in 2012)

Some deviations from the original paper:
- Uses Batch Normalization instead of the custom regularization they used in their paper
- Has an input size of 227x227x3 instead of 224x224x3
- Uses a scaled down version of the ImageNet dataset + fewer epochs of training


To Do:
- Replace ImageNet with a smaller dataset (e.g. Cifar10)
- Write a function for testing accuracy
