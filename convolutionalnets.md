---
title: 
layout: default
---

*previous* - [restricted Boltzmann machine](../restrictedboltzmannmachine.html)
# Convolutional Nets

Convolutional nets take slices of the feature space, say, of an image, and learn them one by one. 

That is, convolutional nets analyze images differently than RBMs. While RBMs learn to reconstruct and identify the features of each image as a whole, convolutional nets learn images in pieces that we call feature maps. 

Picture a grid superimposed on an image, which is broken down into a series of squares. The convolutional net learns each of those squares and then weaves them together in a later stage.

![Alt text](../img/convnet.png) 

By learning different portions of a feature space, convolutional nets allow for easily scalable and robust feature engineering.

### Yann LeCun

[Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/lecun-iscas-10.pdf), a professor at New York University and director of research at Facebook, has done much to advance and promote the use of convolutional nets, which are used heavily in machine vision tasks. 
