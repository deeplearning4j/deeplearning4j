---
title: Demo - VGG-16 Classifier
layout: demo_vgg
---

## VGG-16 in DeepLearning4J

VGG-16 is a convolutional neural network that won the ImageNet Competition in 2015 in the localisation and classification categories. This demo is an example of how Deeplearning4j can work with pre-trained models. This network was imported via Keras using DeepLearning4j's `Keras.importSequential()` feature. 

## Live demo

The form above allows you to upload any image and classify it with the VGG-16 model. 

## Interpreting the result

The demo identifies and returns the top 5 labels associated with the image, and their probabilities. If you submit a picture of yourself, please note that VGG16 was trained to perform facial recognition. The network will certainly fail to identify you, but it will do much better with images of cats and dogs.
 
JSON is returned in the following format:

```
{  
   "data":[  
      {  
         "label":"espresso",
         "prediction":38.31146
      },
      {  
         "label":"cup",
         "prediction":4.0962706
      },
      {  
         "label":"eggnog",
         "prediction":2.588317
      }
   ],
   "performance":{  
      "feedforward":121
   }
}
```

`data` is an array of the predicted labels, and `performance` describes the different timings of the network.

## Pretrained VGG-16 and Transfer Learning

A pre-trained network liked VGG16 is useful for transfer learning. That is, VGG16 has been trained to understand the structure of many images. You specific use case may involve images it hasn't seen before. By swapping out the output layer of VGG16, and training it on your own dataset to distinguish among a new set of labels, you can leverage the training that has already been conducted, and adapt that pre-trained model in less time and less expense (large image-recognition models can cost tens of thousands of dollars to train on popular public cloud services).
