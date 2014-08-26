---
title: 
layout: default
---

*previous* - [mnist for restricted Boltzmann machines](../rbm-mnist-tutorial.html)
# MNIST for deep-belief networks tutorial

MNIST is a good place to begin exploring image recognition. 

To begin with, you’ll take an image from your data set and binarize it, which means you’ll convert its pixels from continuous gray scale to ones and zeros. A useful rule of thumb if that every gray-scale pixel with a value higher than 35 becomes a 1, and the rest are set to 0. The tool you’ll use to do that is an MNIST data-set iterator class.

The [MnistDataSetIterator](../doc/org/datasets/iterator/impl/MnistDataSetIterator.html) does this for you.

A DataSetIterator can be used like this:

         DataSetIterator iter = ....;

         while(iter.hasNext()) {
         	DataSet next = iter.next();
         	//do stuff with the data set
         }

Typically, a DataSetIterator handles inputs and data-set-specific concerns like binarizing or normalization. For MNIST, the following does the trick:
         
         //Train on batches of 10 out of 60000
         DataSetIterator mnistData = new MnistDataSetIterator(10,60000);

The reason we specify the batch size as well as the number of examples is so the user can choose how many examples they want to look at.

Note to Windows uers, in place of the line below, please do the following:

         1. Download the preserialized mnist dataset [https://drive.google.com/file/d/0B-O_wola53IsWDhCSEtJWXUwTjg/edit?usp=sharing](here):

         2. Use the following dataset iterator, this one is equivalent to the one below:    

               DataSet d = new DataSet();
               BufferedInputStream bis = new BufferedInputStream(new FileInputStream(new File("path/to/your/file")));
               d.load(bis);
               bis.close();

          DataSetIterator iter = new ListDataSetIterator(d.asList(),10);

Next, we want to train a deep-belief network to reconstruct the MNIST data set. This is done with following snippet:

<script src="http://gist-it.appspot.com/https://github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/mnist/RawDBNMnistExample.java?slice=24:72"></script>

After your net has trained, you'll see an F1 score. In machine learning, that's the name for one metric used to determine how well a classifier performs. The [f1 score](https://en.wikipedia.org/wiki/F1_score) is a number between zero and one that explains how well the network performed during training. It is analogous to a percentage, with 1 being the equivalent of 100 percent predictive accuracy. It's basically the probability that your net's guesses are correct.

Now that you've seen Deeplearning4j train a neural network on MNIST images, you may want to learn how to deal with continuous data with the [Iris flower dataset](../iris-flower-dataset-tutorial.html).
