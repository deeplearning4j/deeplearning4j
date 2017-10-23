<!-- need to format this into markdown and reader-friendly format, and add NLP chapter -->
# Deeplearning4J Programmers Guide

Plans, outline, and questions for creating the DL4J Programmers Guide.

## Chapter 1. Introduction and Prerequisites

Understanding Deep Learning as a field
	Source. https://deeplearning4j.org/deeplearningforbeginners.html, 
https://deeplearning4j.org/neuralnet-overview
System requirements
	Source.  https://deeplearning4j.org/quickstart#prerequisites 
	CPU/GPU http://nd4j.org/gpu_native_backends.html 
Components --  parts of DL4J, datavec, dl4j, nd4j etc
            Source. https://deeplearning4j.org/overview
	Tools -- list overview
Java, IntelliJ IDE, Maven 
			https://deeplearning4j.org/maven (not in depth enough but a start)
Java vs Python (transitions and mapping comparison)
Link to Troubleshooting, POM dependencies.
https://deeplearning4j.org/troubleshootingneuralnets

## Chapter 2. The Process Overview

Ingesting ETL
Content. General Big picture https://deeplearning4j.org/etl-userguide. Specific piece to document   org.datavec.spark.transform.AnalyzeSpark
Data type overview 
Image (CNN)
General (FF, MLP)
Sequence (LSTM, RNN)
Ingesting Text (Natural Language Processing, NLP)
Processing Data
Spark Transform
Split Test and Train
Shuffle
Labels, Path Label Generator, 
Building NN
MultiLayerNetwork
Computation Graph
Training NN
Using the UI
Evaluating output
Source. org.deeplearning4j.eval.Evaluation
Deploying for Inference
Source.
Building a Neural Network Overview
Load data
Define config
Setup model
Train NN
Validate the model

## Chapter 3. Code Structure

Common Structure by example
Ingesting ETL
Processing Data
Building Multilayer Network
		Content. Training code sample
Training NN
Evaluating output

## Chapter 4. Example: Convolutional Network 

(dense + convolutional + max pooling)
Source.

## Chapter 5. Example: LSTM Network 

(graves LSTM)
Source.

## Chapter 6. Example: FeedForward 

(dense only)
Source.

## Chapter 7. Deploying for Inference

Final piece, pushing it live.

Appendix. Troubleshooting
Appendix. Natural Language Processing


