---
title: Getting started: importing TensorFlow models into SameDiff
short_title: Model import
description: importing TensorFlow models into SameDiff
category: SameDiff
weight: 3
---

# Getting started: importing TensorFlow models into SameDiff

## What models can be imported into SameDiff

Currently SameDiff supports the import of TensorFlow frozen graphs through the various SameDiff.importFrozenTF methods. 
TensorFlow documentation on frozen models can be found [here](https://www.TensorFlow.org/guide/saved_model#the_savedmodel_format_on_disk). 

    import org.nd4j.autodiff.SameDiff.SameDiff;
    
    SameDiff sd = SameDiff.importFrozenTF(modelFile);
    
 ## Finding the model input/outputs and running inference
 
 After you import the TensorFlow model there are 2 ways to find the inputs and outputs. The first method is to look at the output of
 
     sd.summary();
     
 Where the input variables are the output of no ops, and the output variables are the input of no ops.  Another way to find the inputs is
 
      List<String> inputs = sd.inputs();
    
 To run inference use:
 
    INDArray out = sd.batchOutput()
        .input(inputs, inputArray)
        .output(outputs)
        .execSingle();

##  Import Validation.
We have a TensorFlow graph analyzing utility which will report any missing operations (operations that still need to be implemented) [here](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/imports/TensorFlow/TensorFlowImportValidator.java)

## Advanced: Node Skipping and Import Overrides
It is possible to remove nodes from the network. For example TensorFlow 1.x models can have hard coded dropout layers. 
See the [BERT Graph test](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/imports/TFGraphs/BERTGraphTest.java#L114-L150) for an example.

## List of models known to work with SameDiff.
 		
- [PorV-RNN](https://deeplearning4jblob.blob.core.windows.net/testresources/PorV-RNN_frozenmodel.pb)
- [alexnet](https://deeplearning4jblob.blob.core.windows.net/testresources/alexnet_frozenmodel.pb)
- [cifar10_gan_85](https://deeplearning4jblob.blob.core.windows.net/testresources/cifar10_gan_85_frozenmodel.pb)
- [deeplab_mobilenetv2_coco_voc_trainval](http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz)
- [densenet_2018_04_27](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/densenet_2018_04_27.tgz)
- [inception_resnet_v2_2018_04_27](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz)
- [inception_v4_2018_04_27](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz)
- [labels](https://github.com/KonduitAI/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/labels)
- [mobilenet_v1_0.5_128](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_128.tgz)
- [mobilenet_v2_1.0_224](http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz)
- [nasnet_mobile_2018_04_27](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz)
- [resnetv2_imagenet_frozen_graph](http://download.tensorflow.org/models/official/resnetv2_imagenet_frozen_graph.pb)
- [squeezenet_2018_04_27](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz)
- [temperature_bidirectional_63](https://deeplearning4jblob.blob.core.windows.net/testresources/temperature_bidirectional_63_frozenmodel.pb)
- [temperature_stacked_63](https://deeplearning4jblob.blob.core.windows.net/testresources/temperature_stacked_63_frozenmodel.pb)
- [text_gen_81](https://deeplearning4jblob.blob.core.windows.net/testresources/text_gen_81_frozenmodel.pb)