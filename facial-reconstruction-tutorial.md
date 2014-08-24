---
title: 
layout: default
---

*previous* - [Iris flower dataset](../iris-flower-dataset-tutorial.html)
# facial reconstruction

Facial images are one type of continuous data from which deep-learning nets can extract features. In this case, we're using a continuous restricted Boltzmann machine, or CRBM, to identify relevant facial features.

The goal here isn't to classify faces (yet), but to teach the CRBM to reconstruct them based on features shared across faces; i.e. generic traits.  

We're training the net on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/results.html) (LFW) dataset created by UMass/Amherst. LFW contains 13,233 images of 5,749 different faces, so it's fairly large. The dataset is an important and widely used tool in building nets useful for computer vision.

Our network, which you can see [here](https://github.com/agibsonccc/java-deeplearning/tree/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example), learned faces based on a compressed version of those images. The process took about five minutes to run on a reasonably powerful laptop. The results look like this:

![Alt text](../img/LFW_reconstruction.jpg)

One important factor training on these images is *not* to normalize the data. Typically, we would normalize by taking the maximum pixel number of the image, and divide everything by that number. We don't do that here, because we found that normalizing the data hindered learning. Faces are too similar.

The network would pick up on different patterns with more data. This tutorial simply attempts to show  what the network filters look like when actually learning faces; e.g. the reconstructions are not all white noise.

Now that the neural nets know the features that compose faces, the next step will be to classify faces according to distinguishing features. That will be our next tutorial.

For now, the next page in this guide discusses how to feed DL4J other [datasets](../data-sets-ml.html).

Here's some of the code used to run LFW on DL4J:

<script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/lfw/LFWRBMExample.java?slice=25:72"></script>

(For a deeper dive into our LFW code, see this [Github page](https://github.com/agibsonccc/java-deeplearning/tree/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/lfw).)
