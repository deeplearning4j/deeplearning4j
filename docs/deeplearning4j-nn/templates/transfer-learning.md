---
title: Neural Network Transfer Learning
short_title: Transfer Learning
description:
category: Tuning & Training
weight: 5
---

## DL4J’s Transfer Learning API

The DL4J transfer learning API enables users to:

* Modify the architecture of an existing model
* Fine tune learning configurations of an existing model.
* Hold parameters of a specified layer constant during training, also referred to as “frozen" 
 
Holding certain layers frozen on a network and training is effectively the same as training on a transformed version of the input, the transformed version being the intermediate outputs at the boundary of the frozen layers. This is the process of “feature extraction” from the input data and will be referred to as “featurizing” in this document. 


## The transfer learning helper

The forward pass to “featurize” the input data on large, pertained networks can be time consuming. DL4J also provides a TransferLearningHelper class with the following capabilities. 

* Featurize an input dataset to save for future use
* Fit the model with frozen layers with a featurized dataset 
* Output from the model with frozen layers given a featurized input.

When running multiple epochs users will save on computation time since the expensive forward pass on the frozen layers/vertices will only have to be conducted once.


## Show me the code

This example will use VGG16 to classify images belonging to five categories of flowers. The dataset will automatically download from http://download.tensorflow.org/example_images/flower_photos.tgz

#### I.  Import a zoo model

As of 0.9.0 (0.8.1-SNAPSHOT) Deeplearning4j has a new native model zoo. Read about the [deeplearning4j-zoo](/model-zoo) module for more information on using pretrained models. Here, we load a pretrained VGG-16 model initialized with weights trained on ImageNet:

```
ZooModel zooModel = new VGG16();
ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
```


#### II.  Set up a fine-tune configuration

```
FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(5e-5))
            .seed(seed)
            .build();
```

#### III.  Build new models based on VGG16

##### A.Modifying only the last layer, keeping other frozen

The final layer of VGG16 does a softmax regression on the 1000 classes in ImageNet. We modify the very last layer to give predictions for five classes keeping the other layers frozen.

```
ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrainedNet)
    .fineTuneConfiguration(fineTuneConf)
              .setFeatureExtractor("fc2")
              .removeVertexKeepConnections("predictions") 
              .addLayer("predictions", 
        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(4096).nOut(numClasses)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX).build(), "fc2")
              .build();
```
After a mere thirty iterations, which in this case is exposure to 450 images, the model attains an accuracy > 75% on the test dataset. This is rather remarkable considering the complexity of training an image classifier from scratch.

##### B. Attach new layers to the bottleneck (block5_pool)

Here we hold all but the last three dense layers frozen and attach new dense layers onto it. Note that the primary intent here is to demonstrate the use of the API, secondary to what might give better results.

```
ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrainedNet)
              .fineTuneConfiguration(fineTuneConf)
              .setFeatureExtractor("block5_pool")
              .nOutReplace("fc2",1024, WeightInit.XAVIER)
              .removeVertexAndConnections("predictions") 
              .addLayer("fc3",new DenseLayer.Builder()
         .activation(Activation.RELU)
         .nIn(1024).nOut(256).build(),"fc2") 
              .addLayer("newpredictions",new OutputLayer
        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(Activation.SOFTMAX)
                                .nIn(256).nOut(numClasses).build(),"fc3") 
            .setOutputs("newpredictions") 
            .build();
```

##### C. Fine tune layers from a previously saved model 

Say we have saved off our model from (B) and now want to allow “block_5” layers to train. 

```
ComputationGraph vgg16FineTune = new TransferLearning.GraphBuilder(vgg16Transfer)
              .fineTuneConfiguration(fineTuneConf)
              .setFeatureExtractor(“block4_pool”)
              .build();
```

#### IV.  Saving “featurized” datasets and training with them.

We use the transfer learning helper API. Note this freezes the layers of the model passed in.

Here is how you obtain the featured version of the dataset at the specified layer “fc2”.

```
TransferLearningHelper transferLearningHelper = 
    new TransferLearningHelper(pretrainedNet, "fc2");
while(trainIter.hasNext()) {
        DataSet currentFeaturized = transferLearningHelper.featurize(trainIter.next());
        saveToDisk(currentFeaturized,trainDataSaved,true);
  trainDataSaved++;
}
```

Here is how you can fit with a featured dataset. vgg16Transfer is a model setup in (A) of section III.

```
TransferLearningHelper transferLearningHelper = 
    new TransferLearningHelper(vgg16Transfer);
while (trainIter.hasNext()) {
       transferLearningHelper.fitFeaturized(trainIter.next());
}
```

## Notes

* The TransferLearning builder returns a new instance of a dl4j model. 

Keep in mind this is a second model that leaves the original one untouched. For large pertained network take into consideration memory requirements and adjust your JVM heap space accordingly.

* The trained model helper imports models from Keras without enforcing a training configuration. 

Therefore the last layer (as seen when printing the summary) is a dense layer and not an output layer with a loss function. Therefore to modify nOut of an output layer we delete the layer vertex, keeping it’s connections and add back in a new output layer with the same name, a different nOut, the suitable loss function etc etc. 

* Changing nOuts at a layer/vertex will modify nIn of the layers/vertices it fans into. 

When changing nOut users can specify a weight initialization scheme or a distribution for the layer as well as a separate weight initialization scheme or distribution for the layers it fans out to.

* Frozen layer configurations are not saved when writing the model to disk. 

In other words, a model with frozen layers when serialized and read back in will not have any frozen layers. To continue training holding specific layers constant the user is expected to go through the transfer learning helper or the transfer learning API. There are two ways to “freeze” layers in a dl4j model.

    - On a copy: With the transfer learning API which will return a new model with the relevant frozen layers
    - In place: With the transfer learning helper API which will apply the frozen layers to the given model.

* FineTune configurations will selectively update learning parameters. 

For eg, if a learning rate is specified this learning rate will apply to all unfrozen/trainable layers in the model. However, newly added layers can override this learning rate by specifying their own learning rates in the layer builder.

## Utilities

{{autogenerated}}