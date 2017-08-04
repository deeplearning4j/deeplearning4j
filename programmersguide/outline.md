# Deeplearning4J Programmers Guide

Contents

* [Introduction and Prerequisites](#prereqs)
* [The Process Overview](#processoview)
* [Code Structure](#codestructure)
* [Example: Convolutional Network](#cnn)
* [Example: LSTM Network](#lstm)
* [Example: Recurrent Network](#rnn)
* [Example: FeedForward](#feedforward)
* [Example: Dense Layer](#denselayer)
* [Deploying for Inference](#inference)
* [Troubleshooting](#troubleshooting)
* [Natural Language Processing](#nlp)
### Sub-Head Title1
`code text`

DeepLearning for Java, (deeplearning4j). 

## Chapter 1. <a name=“prereqs”>Introduction and Prerequisites</a>

### Understanding DeepLearning as a field

Source. https://deeplearning4j.org/deeplearningforbeginners.html, https://deeplearning4j.org/neuralnet-overview

### System requirements	

Source.  https://deeplearning4j.org/quickstart#prerequisites 	

CPU/GPU http://nd4j.org/gpu_native_backends.html 

### Components

parts of DL4J, datavec, dl4j, nd4j etc            

Source. https://deeplearning4j.org/overview	

### Tools

list overview — Java, IntelliJ IDE, Maven 			

https://deeplearning4j.org/maven (not in depth enough but a start)

Java vs Python

 (transitions and mapping comparison)

Link to Troubleshooting, POM dependencies.

https://deeplearning4j.org/troubleshootingneuralnets

## Chapter 2. <a name=“processoview”>The Process Overview</a>

### Ingesting ETL

Content. General Big picture https://deeplearning4j.org/etl-userguide. 

Specific piece to document   

org.datavec.spark.transform.AnalyzeSpark

#### Data type overview 

Image (CNN)

General (FF, MLP)

Sequence (LSTM, RNN)

Ingesting Text (Natural Language Processing, NLP)

### Processing Data

Spark Transform

Split Test and Train

Shuffle

Labels, Path Label Generator

### Building NN

MultiLayerNetwork

Computation Graph

### Training NN

Using the UI

### Evaluating output

Source. org.deeplearning4j.eval.Evaluation

### Deploying for Inference

Source.

Building a Neural Network Overview

Load data

Define config

Setup model

Train NN

Validate the model

## Chapter 3. <a name=“codestructure”>Code Structure</a>

- Common Structure by example
- Feed Forward or MLP 

MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed) //for reproducabilty
        .iterations(iterations)// how often to update, see epochs and minibatch
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .learningRate(learningRate)
        .updater(Updater.NESTEROVS).momentum(0.9)
        .regularization(true).l2(1e-4)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.TANH)
        .list()
        .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                .build())
        .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                .build())
        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(numHiddenNodes).nOut(numOutputs).build())
        .pretrain(false).backprop(true).build();
Convolutional
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .seed(rngseed)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .iterations(1)
    .learningRate(0.006)
    .updater(Updater.NESTEROVS).momentum(0.9)
    .regularization(true).l2(1e-4)
    .list()
        .layer(0, new ConvolutionLayer.Builder(5, 5)
                //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                .nIn(channels)
                .stride(1,1)
                .nOut(20)
                .activation(Activation.IDENTITY)
                .build())
        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .build())
        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .build())
        .pretrain(false).backprop(true)
    .setInputType(InputType.convolutional(height,width,channels))
    .build();
- Ingesting ETL

- Processing Data

   - Building Multilayer Network

 Content. Training code sample


- Training NN
- Evaluating output

##Example: <a name=“cnn”>Convolutional Network</a>

(dense + convolutional + max pooling)

Source.

##Example: <a name=“lstm”>LSTM Network</a>

(graves LSTM)

Source.

##Example: <a name=“rnn”>Recurrent Network</a>

Source.

##Example: <a name=“feedforward”>FeedForward</a>

(dense only)

Source.

##Example: <a name=“denselayer”>Dense Layer</a>

Source.

##Chapter 9. <a name=“inference”>Deploying for Inference</a>

Final piece, pushing it live.

##Appendix. <a name=“troubleshooting”>Troubleshooting</a>

##Appendix. <a name=“nlp”>Natural Language Processing</a>