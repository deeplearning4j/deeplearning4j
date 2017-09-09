# Deeplearning4J Programmers Guide

Contents

* [Introduction and Process](#process)
* [Requirements and Components](#prereqs)
* [Ingestion: DataVec](#datavec)
* [Using Maven and IntelliJ](#tools)
* [Code Structure](#codestructure)
* [Example: Convolutional Network](#cnn)
* [Example: LSTM Network](#lstm)
* [Example: FeedForward](#feedforward)
* [Deploying for Inference](#inference)
* [Troubleshooting](#troubleshooting)
* [Natural Language Processing](#nlp)

## Chapter 1. <a name=“process”>Introduction and Process</a>

### Understanding DeepLearning as a field

Source. https://deeplearning4j.org/deeplearningforbeginners.html, https://deeplearning4j.org/neuralnet-overview

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

## Chapter 2. <a name=“prereqs”>Requirements and Components</a>
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



## Chapter 3. <a name=“codestructure”>Code Structure</a>

Common Structure by example

- Feed Forward or MLP 

  MultiLayerConfiguration conf = new    NeuralNetConfiguration.Builder()

- Convolutional

  MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()

- Building Multilayer Network


- Training NN code sample

  Deconstructed

## Chapter 4. <a name="datavec">Ingestion : DataVec</a>

Datavec portion of the examples.

ETL guide

Ingesting ETL

Processing Data

## Chapter 5. <a name="tools">Using Maven and IntelliJ</a>
Start and retrieve DL4J examples.

## Example: <a name=“cnn”>Convolutional Network</a>

(dense + convolutional + max pooling)

Source. MNIST, heavy model — imported, VGG or Inception

Tom has a MNIST sample. 

Most deply many layers with many combinations. 

Ingestion requirement 

## Example: <a name=“lstm”>LSTM Network</a>

(graves LSTM)

Source. Shakespeare, Audio files > Midi sequence generator

Recurrent Network Source. Merge with LSTM, 

## Example: <a name=“feedforward”>FeedForward</a>

(dense only)

Source.

Dense Layer Source. Merge with FeedForward. 

Simple to explain concept. IRIS. 


## Chapter 9. <a name=“inference”>Deploying for Inference</a>

Final piece, pushing it live.

## Appendix. <a name=“troubleshooting”>Troubleshooting</a>

## Appendix. <a name=“nlp”>Natural Language Processing</a>
