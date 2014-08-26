---
title: 
layout: default
---

*previous* - [mnist for deep-belief networks](../mnist-tutorial.html)
# Iris Flower Dataset Tutorial

The Iris flower dataset is widely used in machine learning to test classification techniques. We will use it to verify the effectiveness of our neural nets. 

The dataset consists of four measurements taken from 50 examples of each of three species of Iris, so 150 flowers and 600 data points in all. The length and width of both sepals and petals were measured for the species Iris setosa, Iris virginica and Iris versicolor. 

The continuous nature of those measurements make the Iris dataset a perfect test for continuous deep-belief networks. Those four features alone can be sufficient to classify the three species accurately, and failure to do so is a very strong signal that your neural net needs work.

While the dataset is small, which can present its own problems, the species virginica and versicolor are so similar that they partially overlap. That is, this set will make you work to classify it properly.

DL4J's neural nets can classify the set within two minutes with an accuracy of greater than 90 percent. Once you've seen DL4J train a neural network on Iris, you may want to learn about [facial reconstruction](../facial-reconstruction-tutorial).

![Alt text](../img/iris_dataset.png)

The code to run the Iris dataset on DL4J looks like this:

<script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/iris/IrisRBMExample.java?slice=16:32"></script>

To run the Iris dataset on Deeplearning4j, you can either click "run" on the IrisRBMExample Java file in IntelliJ (see our [Getting Started page](../gettingstarted.html)), or type this into your command line:

    java -cp "lib/*" org.deeplearning4j.example.iris.IrisRBMExample

After your net has trained, you'll see an F1 score. In machine learning, that's the name for one metric used to determine how well a classifier performs. The [f1 score](https://en.wikipedia.org/wiki/F1_score) is a number between zero and one that explains how well the network performed during training. It is analogous to a percentage, with 1 being the equivalent of 100 percent predictive accuracy. It's basically the probability that your net's guesses are correct.

[For a deeper dive, see our Iris code on Github](https://github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/iris/IrisExample.java).
