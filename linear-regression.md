---
title: 
layout: default
---

# Neural Networks for Linear Regression

Broadly speaking, neural networks are used for clustering and classification. They cluster unlabeled data or classify data after supervised training. 

But they also have a third use case that's not frequently discussed: *linear regression*. 

While classification typically converts continuous data into dummy variables like 0 and 1 to apply a label -- e.g. given someone's height, weight and age you might bucket them as a  heart-disease candidate or not -- linear regression maps one set of continuous data to continuous output. 

For example, given the age and floor space of a house and its distance from a good school, you might predict what the house would sell for. No dummy variables, just mapping independent variables x to a continuous y.

Reasonable people can disagree about whether using neural networks for linear regression is overkill. The point of this post is just to show that it can be done, and in fact, it's pretty easy.

![Alt text](../img/neural-network-linear-regression.png)

In the diagram above, x stands for input, the features passed forward from the network's previous layer. Many x's will be fed into each node of the last hidden layer, and each x will be multiplied by a corresponding weight, w.

The sum of those products is added to a bias and fed into an activation function. In this case the activation function is a rectified linear unit (ReLU), commonly used and highly useful because it doesn't saturate on shallow gradients as sigmoid activation functions do.
 
For each hidden node, ReLU outputs an activation, a, and the activations are summed going into the output node, which simply passes the activations' sum through. 

That is, a neural network performing linear regression will have one output node, and that node will just multiply the sum of the previous layer's activations by 1. The result will be ŷ, "y hat", the network's estimate, the dependent variable that all your x's map to. 

To perform backpropagation and make the network learn, you simply compare ŷ to the ground-truth value of y and adjust the weights and biases of the network until error is minimized, much as you would with a classifier. 

In this way, you can use a neural network to get the function relating an arbitrary number of independent variables x to a dependent variable y that you're trying to predict. 

### Other Resources

* [Introduction to Deep Neural Networks](../neuralnet-overview.html)
* [Iris Tutorial](../iris-flower-dataset-tutorial.html)
* [Deeplearning4j Quickstart Examples](../quickstart.html)
