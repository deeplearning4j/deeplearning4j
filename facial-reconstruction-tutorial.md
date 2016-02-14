---
title: Facial reconstruction
layout: default
---

# Facial reconstruction

Facial images are one type of continuous data from which deep-learning nets can extract features. In this case, we're using a deep-belief network, or DBN, to identify relevant facial features.

The goal here isn't to classify faces (yet), but to teach the DBN to reconstruct them based on features shared across faces; i.e. generic traits.  

We're training the net on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/results.html) (LFW) dataset created by UMass/Amherst. LFW contains 13,233 images of 5,749 different faces, so it's fairly large. The dataset is an important and widely used tool in building nets useful for computer vision.

Our net, which you can see [here](https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/deepbelief/DBNLFWExample.java), learned faces based on a compressed version of those images. The process took about five minutes to run on a reasonably powerful laptop. The results look like this:

![Alt text](../img/LFW_reconstruction.jpg)

One important factor training on these images is *not* to normalize the data. Typically, we would normalize by taking the maximum pixel number of the image, and divide everything by that number. We don't do that here, because we found that normalizing the data hindered learning. Faces are too similar.

The network would pick up on different patterns with more data. This tutorial simply attempts to show  what the network filters look like when actually learning faces; e.g. the reconstructions are not all white noise.

Now that the neural nets know the features that compose faces, the next step will be to classify faces according to distinguishing features. That will be our next tutorial.

Here's some of the code used to run LFW on DL4J:

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/convolution/CNNLFWExample.java?slice=35:111"></script>

To run the Labeled Faces in the Wild dataset on Deeplearning4j, you can either click "run" on the DBNLFWExample.java file in IntelliJ (see our [**Getting Started** page](../gettingstarted.html)).

After your net has trained, you'll see an F1 score. In machine learning, that's the name for one metric used to determine how well a classifier performs. The [f1 score](https://en.wikipedia.org/wiki/F1_score) is a number between zero and one that explains how well the network performed during training. It is analogous to a percentage, with 1 being the equivalent of 100 percent predictive accuracy. It's basically the probability that your net's guesses are correct.

For a deeper dive into our LFW code, see this [Github page](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/convolution/CNNLFWExample.java).
