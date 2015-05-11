---
title: 
layout: default
---

# MNIST for Deep-Belief Networks

MNIST is a good place to begin exploring image recognition. The first step is to take an image from the dataset and binarize it; i.e. convert its pixels from continuous gray scale to ones and zeros. Typically, every gray-scale pixel with a value higher than 35 becomes a 1, while the rest are set to 0. The MNIST dataset iterator class does that.

An **MnistDataSetIterator**, just a form of a more general DataSetIterator, will do that. You can find it used in this example:

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/mnist/full/DBNExample.java?slice=50:59"></script>

Typically, a DataSetIterator handles inputs and dataset-specific concerns like binarization or normalization. For MNIST, the following line specifies the batch size and number of examples, two parameters which allow the user to specify the sample size they want to train on (more examples tend to increase both model accuracy and training time):
         
         //Train on batches of 100 out of 60000 examples
         DataSetIterator iter = new MnistDataSetIterator(100,60000);

Note to Windows users, in place of the line below, please do the following:

1. Download the preserialized Mnist dataset [here](https://drive.google.com/file/d/0B-O_wola53IsWDhCSEtJWXUwTjg/edit?usp=sharing).
2. Use the following dataset iterator, this one is equivalent to the one below:    

       DataSet d = new DataSet();
       BufferedInputStream bis = new BufferedInputStream(new FileInputStream(new File("path/to/your/file")));
       d.load(bis);
       bis.close();
        
       DataSetIterator iter = new ListDataSetIterator(d.asList(),10);

Next, we want to train a deep-belief network to reconstruct the MNIST dataset. Here's how you configure your deep-belief network (by giving it three **hiddenLayerSizes**, we have effectively created three hidden layers):

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/mnist/full/DBNExample.java?slice=37:50"></script>

Notice there is no DBN class, just a stack of [restricted Boltzmann machines (RBMs) discussed  here alongside their parameters](http://deeplearning4j.org/restrictedboltzmannmachine.html). 

In the above example, we have defined the momentum, learning rate, made conjugate gradient the optimization algorithm and reconstruction cross entropy the loss function. Weights are initialized in a normal distribution.  

**nIn** is the size of the input data. The sample of grayscale pixels that the net interprets to classify each image. 
**nOut** is the number of classifications -- 10 -- because we are dealing with the numerals 0-9.

After your net has trained, you'll see a a number between zero and one called an [f1 score](https://en.wikipedia.org/wiki/F1_score). In machine learning, that's a metric used to determine how well a classifier performs. It is analogous to a percentage, with 1 being the equivalent of 100 percent predictive accuracy, and you can interpret it as the probability that your net's guesses are correct.

Now that you've seen a neural network train on MNIST images, learn how to train on continuous data with the [Iris flower dataset](../iris-flower-dataset-tutorial.html).
