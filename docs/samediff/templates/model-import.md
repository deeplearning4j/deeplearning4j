---
title: Getting started: importing TensorFlow and ONNX models into SameDiff
short_title: Model import
description: importing TensorFlow and ONNX models into SameDiff
category: SameDiff
weight: 3
---

# Getting started: importing TensorFlow models into SameDiff

## What models can be imported into samediff

Currently samediff supports the import of Tensorflow frozen graphs through the various Samediff.importFrozenTF methods. 
Tensorflow documentation on frozen models can be found [here](https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk). 

    import org.nd4j.autodiff.samediff.SameDiff;
    
    SameDiff sd = SameDiff.importFrozenTF(modelFile);
    
 ## Finding the model input/outputs and running inference
 
 After you import the Tensorflow model there are 2 ways to find the inputs and outputs. The first method is to look at the output of
 
     sd.summary();
     
 Where the input variables are the output of no ops, and the output variables are the input of no ops.  The other way to find the inputs and outputs is
 
      List<String> inputs = sd.inputs();
      List<String> outputs = sd.outputs());
    
 To run inference use:
 
    INDArray out = sd.batchOutput()
        .input(inputs, inputArray)
        .output(outputs)
        .execSingle();

##  Node skipping, import overrides and model validation.

It is possible to remove nodes from the network. For example Tensorflow 1.x models can have hard coded dropout layers. 
See the [BERT Graph test](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/imports/TFGraphs/BERTGraphTest.java#L114-L150) for an example.
We have a Tensorflow graph analyzing utility which will report any missing operations (operations that still need to be implemented) [here](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/imports/tensorflow/TensorFlowImportValidator.java)


## List of models known to work with Samediff.
 		
- [PorV-RNN](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/PorV-RNN)
- [alexnet](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/alexnet)
- [cifar10_gan_85](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/cifar10_gan_85)
- [deeplab_mobilenetv2_coco_voc_trainval](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/deeplab_mobilenetv2_coco_voc_trainval)
- [densenet_2018_04_27](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/densenet_2018_04_27)
- [inception_resnet_v2_2018_04_27](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/inception_resnet_v2_2018_04_27)
- [inception_v4_2018_04_27](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/inception_v4_2018_04_27)
- [labels](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/labels)
- [mobilenet_v1_0.5_128](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/mobilenet_v1_0.5_128)
- [mobilenet_v2_1.0_224](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/mobilenet_v2_1.0_224)
- [nasnet_mobile_2018_04_27](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/nasnet_mobile_2018_04_27)
- [resnetv2_imagenet_frozen_graph](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/resnetv2_imagenet_frozen_graph)
- [squeezenet_2018_04_27](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/squeezenet_2018_04_27)
^ [temperature_bidirectional_63](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/temperature_bidirectional_63)
- [temperature_stacked_63](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/temperature_stacked_63)
- [text_gen_81](https://github.com/deeplearning4j/dl4j-test-resources/tree/master/src/main/resources/tf_graphs/zoo_models/text_gen_81)
