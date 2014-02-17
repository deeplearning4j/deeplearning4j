---
title: 
layout: default
---

# mnist

The MNIST database is a large set of handwritten digits that are used to train neural networks and other algorithms in image recognition. MNIST has 60,000 images in its training set and 10,000 in its test set. 

MNIST derives from NIST, and stands for “Mixed National Institute of Standards and Technology.” The MNIST database reshuffles the NIST database's thousands of binary images of handwritten digits in order to better train and test various image recognition techniques. A full explanation of why MNIST is preferable to NIST can be found on [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/).

Each image in the MNIST database is a 28x28 pixel cell, and each cell is contained within a bounding box, four lines of pixels to frame it. The image is centered according to the center of mass of its pixels. 

MNIST is a good place to begin exploring image recognition. Here’s an easy way to load the data and get started. 

### tutorial

To begin with, you’ll take an image from your [training?] data set and binarize it, which means you’ll convert its pixels from continuous gray scale to ones and zeros. A useful rule of thumb if that every gray-scale pixel with a value higher than 35 becomes a 1, and the rest are set to 0. The tool you’ll use to do that is an MNIST data-set iterator class.

CODE HERE

Now you should have a matrix of ones and zeros, and that’s the input you’ll feed into your neural network. A restricted Boltzmann machine will look at that input and learn patterns among the ones and zeros that correlate with significant features; i.e. features that lead to lower erros. 

Next, you deliberately corrupt the data. You’re basically throwing noise at it for the same reason that denoising autoencoders do. You need to prove to yourself that your net is not simply reproducing the input as output, a pitfall known as the identity function. Only when the algorithm is capable of taking bad data and ignoring the noise can it be trusted to learn unsupervised. 

Restricted Boltzmann machines learn by executing a repeated steps of contrastive divergence; i.e. they take stochastic steps through various combinations of coefficient weights to test which ones lead it closest to its benchmark. 

That ensemble of steps is called conjugate gradient, which is a way the machine has of determining when it’s done. When error ceases to diminish, and starts to increase, it has overshot its local optimum. 

When the training on one data cell is complete, you will be shown the image it has reconstructed, based on the features it selected as relevant. That image ought to resemble a real number. Then it moves on to the next