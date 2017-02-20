## VGG-16 in DeepLearning4J

VGG-16 is a convolutional neural network that won the ImageNet Competition in 2015 in the localisation and classification categories. This Network was created by taking a pretrained network and the trained weights and importing it using DeepLearning4J's
`Keras.importSequential()` feature. 

## Live demo

The form above allows you to upload any image to a Deeplearning4j and classify it with the VGG-16 model.  

## Interpreting the result

The demo identifies and returns the top 5 labels and their probabilities. If you submit a picture of yourself, please note that this competition was not for facial recognition. The network will certainly fail to identify you, images of cats and dogs it will do much better with.
 
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

Modifying this trained network for your use case would involve either "transfer learning" a feature we are working on, or loading the model and then performing additional training on your dataset. 



