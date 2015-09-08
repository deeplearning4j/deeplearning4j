---
title: 
layout: default
---

# Saving and Loading a Neural Network

There are two ways to save a model with Deeplearning4j. The first is Java serialization. Neural networks that implement the Persistable interface are persistable via Java serialization. It looks like this in the [Javadoc](http://deeplearning4j.org/doc/), where you can find it by searching for "Persistable."

![Alt text](../img/persistable.png) 

With this binary, any neural network can be saved and sent across a network. Deeplearning4j uses it for the distributed training of neural nets. 

![Alt text](../img/datasets.png) 

It allows neural networks to be loaded into memory afterward (to resume training), and can be used with a REST API or other tools.

## <a name="vector">Save an Interoperable Vector of All Weights</a>

The second format for saving a model is as a long vector of all coefficients, while writing your configuration to a JSON file. 

    INDArray params = layer.params(); 
    //^^turn model's weight matrix into vector
    String conf = layer.conf().toJson(); 
    
    Nd4j.write(params,"somefile.json"); 
    //^^save vector as a string to file type of your choice, or
    Nd4j.writeTxt(params,"somefile.json"); 
    
    
    //You can reload the conf with
    MultiLayerConfiguration = MultiLayerConfiguration.fromJson("somefile.json");
    //or
    NeuralNetConfiguration = NeuralNetConfiguration.fromJson("somefile.json"); 
    
    //You can set the params with 
    layer.setParams(params);
    //or
    multiLayerNetwork.setParams(params);
