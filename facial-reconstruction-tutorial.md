---
title: 
layout: default
---

*previous* - [Iris flower dataset](../iris-flower-dataset-tutorial.html)
# facial reconstruction

Facial images are one type of continuous data deep-learning nets can extract features from. In this case, we're using a continuous restricted Boltzmann machine, or CRBM, to identify relevant facial features.

The goal here isn't to classify faces, yet, but to teach the CRBM to reconstruct them based on features shared across faces -- generic traits.  

We're training the net on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/results.html) (LFW) dataset created by UMass/Amherst. LFW contains 13,233 images of 5,749 different faces, so it's fairly large. The dataset is an important tool in building nets useful for computer vision.

Our network learned faces based on a compressed version of those images. The process took about five minutes to run on a reasonably powerful laptop. The results look like this:

![Alt text](../img/LFW_reconstruction.jpg)

One important factor training on these images is *not* to normalize the data. Typically, we would normalize by taking the maximum pixel number of the image, and dividing everything by that number. We don't do that here, because there is not enough variance in the data. Faces are too similar.

Now that the neural nets knows the features that compose faces, the next step will be to classify faces according to distinguishing features. That will be out next page.

For now, the next step in this guide is about how to feed DL4J other [datasets](../data-sets-ml.html).

