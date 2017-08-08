title: DeepLearning4j Overview and Process
layout: default 

------

# Where DeepLearning4j Fits

(https://deeplearning4j.org/deeplearningforbeginners.html, https://deeplearning4j.org/neuralnet-overview)

DeepLearning4j is a Java based toolkit for building, training, and deploying Neural Networks.  

Typically, there are few reasons why you might be interested in DeepLearning4j. 

- As a data scientist in the field or student with a Java project. Typical use cases allow for experiments and continuous training on clusters to manage processes for the most accruate model over the lifetime of the project. 

- As a data engineer or developer in an enterprise environment and tasked with providing reliable, scalable, accurate conclusions across a structured organization. The use case here is to have data programmatically and automatically processed and analyzed to determine a designated result, using simple and understandable APIs, leveraging day to day activities without impact to business operations.


Hiearchically, DeepLearning4J is a type of Neural Network model within the deep learning field, that is a subset to machine learning, which in turn is a subset of artifical intelligence. Broadly, it is a form of data mining, extracting and deducing information from a set of data. 

Neural Networks are a computational model, containing multi-dimensional arrays of numerical data, using algorithms to extract information from the raw data, so the model can be used to return inferences about new data that has not yet been processed. 

You build the Neural Network to learn about the type of data, and to return consistent, accurate, interpretations of the data. Once it has trained sufficiently, it can be used to process new sets of data to return reasonably reliable conclusions and inferences from the data. 

The complete process requires the following phases. DeepLearning4j has a role in each phase.

1. Preparing the data
  - Ingesting the data, ETL
    - Extract from the data source 
    - Transform to arrays of numbers
    - Load the data
  - Reviewing the data types
  - Processing the data
2. Preparing the Neural Network
  - Building the Neural Network
    - Define configuration
    - Define dependencies, pom.xml
    - Setup the model
    - Validate the model
  - Training the Neural Network
3. Evaluating, Adjusting, and Deploying
  - Evaluating the output
  - Repeating step 2 until the output meets the goals
  - Deploying the Neural Network

# DeepLearning4j Process

The complete process requires the following phases. At each phase, you apply decisions based on the source material and the target results. 

## Framing the Question

Identify the source content provided and the conclusions expected. What is the relationship between the source and the conclusion? 

- Identify based on characteristics
- Recommender based on previous
- Predict outcome based on conditions

# Preparing the Data

The type of source data you feed the Neural Network affects the type of algorithms you apply to analyze the data. To prepare a model to analyze data, need relevant data.

Identify the source data. Is the data images, sounds, text, or numbers? Is time relevant to either collecting the source data or the target result?

Content. General Big picture https://deeplearning4j.org/etl-userguide.  https://deeplearning4j.org/datavecdoc/org/datavec/spark/transform/AnalyzeSpark.html

## Data Type Overview

- Image (CNN)
- General (FF, MLP)
- Sequence (LSTM, RNN)
- Ingesting Text (Natural Language Processing, NLP)

## Using DataVec: DeepLearning4j Data Ingesting Tool

(https://deeplearning4j.org/overview#datavec)

Transforming data into numbers. 

DataVec is a set of DeepLearning4j tools for turning raw data, such as images, video, audio, text and time series, into feature vectors for Neural Networks. Use these tools for ingesting, cleaning, joining, scaling, normalizing and transforming the data prior to performing any sort of analysis. 

## Processing Data

Track and read meta data from arrays of rows and columns. 

Preprocessor for NN, features that will be in NN, track and read meta data what row/column. Collecting from various data sources, ex 2 files systems on disk somewhere and databases somewhere and columns from db that were public, create mixed, >> normalize it, preprocess >> all helped by dataVec. in a program that is coherent to ETL. 

That preparation includes non-trivial scientific reasoning. Not just selecting data this or that column, but decide analyzing flower, ex Iris. classify based on sepal and petal - put model on the data, model doesn’t know what these are. ex what is important - size b/w sepal and petal not relevant. Since know queries, sub species unknown, which iris is this given a dandelion. 

Relative sepal and relative petal length, rather larger than ave petal or sepal length. To represent > standard normalization. Different b/t petal length > standard deviation. where that example is wrt standard deviation. 

- Spark Transform
- Split Test and Train
- Shuffle
- Labels, Path Label Generator

# Building and Training the Neural Network

Select a type of Neural Network and algorithm appropriate to the data and target return. This is the phase where you design your Neural Networks. 

## Common Neural Networks Overview

- Convoutional Network — images.
- Long Short Term Memory (LSTM) Networks — sequence.
- Recurrent Networks — sequence. Type of LSTM.
- FeedForward Networks — clasification. Example by age or size. A correlation might not exist. 
- Dense Layered Networks — classification. Type of FeedForward.

Sequence classification — for trending patterns, stable, step up/down.

Multi-variant time series — 

## Using DeepLearning4j for Neural Network Models

(https://deeplearning4j.org/overview#deeplearning4j)

DeepLearning4j is a Java-based Neural Network tool designed for multi-layered configurations. 

Two critical understanding about using Java:

- Java uses a nested structure.
- The pom.xml file describes the dependencies for the process. 

## Building Neural Network

  Multi-Layer Network

  Computation Graph

## Declare Dependencies in the pom.xml File

https://deeplearning4j.org/quickstart, Using DL4J In Your Own Projects: Configuring the POM.xml File

Java requires specific dependencies be defined. Not defining these is the number one cause for errors. 

1. Using Maven, locate the pom.xml file in the https://github.com/deeplearning4j/deeplearning4j repository.
2. Open the file for editing.
3. Locate and edit the dependencies for the training model you are going to use. 
   Edit the fields to identify your test resources. 
   - groupId
   - artifactId
   - version
   - scope
     If you are using nd4j-native or nd4j-cuda, edit the corresponding fields for those dependencies as well. 
     Sample `pom.xml` dependencies. 
         <profile>
            <id>testresources</id>
            <activation>
                <activeByDefault>false</activeByDefault>
            </activation>
            <dependencies>
                <dependency>
                    <groupId>org.deeplearning4j</groupId>
                    <artifactId>dl4j-test-resources</artifactId>
                    <version>${dl4j-test-resources.version}</version>
                    <scope>test</scope>
                </dependency>
            </dependencies>
         </profile>

# Evaluating, Adjusting, and Deploying

[NEED DATA]

Evaluating output

Source. (https://deeplearning4j.org/doc/org/deeplearning4j/eval/Evaluation.html)

Keras model import functionality. 


