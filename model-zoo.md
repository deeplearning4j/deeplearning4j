---
title: Deeplearning4j Model Zoo
layout: default
---

# Deeplearning4j Model Zoo

Similar to Caffe's model zoo, this repository provides common model configurations. 

Currently, it contains computer vision configurations:

* [AlexNet](https://github.com/deeplearning4j/dl4j-model-z/blob/master/src/main/java/org/deeplearning4j/AlexNet.java)
* [LeNet](https://github.com/deeplearning4j/dl4j-model-z/blob/master/src/main/java/org/deeplearning4j/LeNet.java)
* [VGGNetA](https://github.com/deeplearning4j/dl4j-model-z/blob/master/src/main/java/org/deeplearning4j/VGGNetA.java)
* [VGGNetD](https://github.com/deeplearning4j/dl4j-model-z/blob/master/src/main/java/org/deeplearning4j/VGGNetD.java)
* [GoogleLeNet](https://github.com/deeplearning4j/dl4j-model-z/blob/master/src/main/java/org/deeplearning4j/GoogleLeNet.java)

The configurations will give you a headstart when creating models for image recognition. 

Deeplearning4j can now import models from Keras, and from other major frameworks such as Theano, TensorFlow, Caffe and Torch via Keras.

* [InceptionV3 image recognition pretrained on ImageNet](https://github.com/USCDataScience/dl4j-kerasimport-examples/tree/master/dl4j-import-example)
* [VGG16](https://github.com/deeplearning4j/deeplearning4j/blob/b69439b3554698390533e8d05f235dfef1195df3/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/trainedmodels/TrainedModels.java)

Instructions for [importing models from Keras are here](https://deeplearning4j.org/model-import-keras).
