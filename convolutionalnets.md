---
title: 
layout: default
---

# Convolutional Nets

Convolutional nets perform object recognition with images. They can identify faces, individuals, street signs, eggplants, platypuses, and many other aspects of visual data. Convolutional nets overlap with text analysis via optical character recognition, but they are also useful when analyzing words as discrete textual units, as well as sound. 

The efficacy of convolutional nets (ConvNets) in image recognition is one of the main reasons why the world has woken up to the power of deep learning. They are powering major advances in machine vision, which has obvious applications for self-driving cars, robotics, drones, and treatments for the visually impaired. 

Convolutional nets take slices of the feature space, say, of an image, and learn them one by one. By learning different portions of a feature space, convolutional nets allow for easily scalable and robust feature engineering.

Note that convolutional nets analyze images differently than RBMs. While RBMs learn to reconstruct and identify the features of each image as a whole, convolutional nets learn images in pieces that we call feature maps. 

Convolutional networks perform a sort of search. Picture a small window sliding left to right across a larger image, and recommencing at the left once it reaches the end of one pass (like typewriters do). That moving window is capable recognizing only one thing, say, a short vertical line. Three dark pixels stacked atop one another. It moves that vertical-line-recognizing filter over the actual pixels of the image, looking for matches.

Each time a match is found, it is mapped onto a feature space particular to that visual element. In that space, the location of each vertical line match is recorded, a bit like birdwatchers leave pins in a map to mark where they last saw a great blue heron. A convolutional net runs many, many searches over a single image – horizontal lines, diagonal ones, as many as there are visual elements to be sought. 

![Alt text](../img/convnet.png) 

Those signals are then passed through a nonlinear transform such as tanh to the second major stage of convolutional nets: pooling, which has two flavors: max or average. The pooling aggregates the feature maps (subsections of subsections) onto one space to get an overall “expectation” of where features occur. This expectation is then projected onto a 2D space relative to the hidden layer size of the convolutional layer.

With max pooling, only the locations on the image that showed the strongest correlation to the feature (the maximum value) are preserved. Much information about lesser values is lost in this step, also known as downsampling, and that has spurred research into alternative methods. But downsampling has the advantage, precisely because information is lost, of decreasing the amount of storage required by the net. 

Convolutional neural networks are related to signal processing. In signal processing, a convolution is the mathematical product of two functions reflecting their overlap, or correlation. Think of a convolution as a way of mixing two inputs by multiplying them. 

### Yann LeCun

[Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/lecun-iscas-10.pdf), a professor at New York University and director of research at Facebook, has done much to advance and promote the use of convolutional nets, which are used heavily in machine vision tasks. 

[Here's our Github test for ConvNets](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/models/layers/ConvolutionDownSampleLayerTest.java).

 <script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/models/layers/ConvolutionDownSampleLayerTest.java?slice=55:99"></script>
