---
title: 
layout: default
---

*previous* - [mnist for deep-belief networks](../mnist-tutorial.html)
# iris flower dataset tutorial

The Iris flower dataset is widely used in machine and deep learning to test classification techniques. We will use it to verify the effectiveness of our neural nets. 

The dataset consists of four measurements taken from 50 examples of each of three species of Iris, so 150 flowers and 600 data points in all. The length and width of both sepals and petals were measured for the species Iris setosa, Iris virginica and Iris versicolor. 

The continuous nature of those measurements make the Iris dataset a perfect test for continuous deep-belief networks. Those four features alone can be sufficient to classify the three species accurately, and failure to do so is a very strong signal that your neural net needs work.

While the dataset is small, which can present its own problems, the species virginica and versicolor are so similar that they partially overlap. That is, this set will make you work to classify it properly.

DL4J's neural nets can classify the set within two minutes with an accuracy of greater than 90 percent. Once you've seen DL4J train a neural network on Iris, you may want to learn how to feed it other [datasets](../data-sets-ml.html).