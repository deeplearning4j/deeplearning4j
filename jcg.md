---
title: jcg
layout: default
---

# Introduction to Neural Networks

Deep learning encompasses both deep neural networks and deep reinforcement learning, which are subsets of machine learning, which itself is a subset of artifical intelligence. Broadly speaking, deep neural networks perform machine perception that extracts important features from raw data and makes some sort of prediction about each observation. Examples include identifying objects represented in images, mapping analog speech to written transcriptions, categorizing text by sentiment, and making forecasts about time series data.

Although neural networks were invented last century, only recently have they generated more excitement. Now that the computing ability to take advantage of the idea of neural networks exists, they have been used to set new, state of the art results in such fields as computer vision, natural language processing, and reinforcement learning. One well known accomplishment of deep learning was achieved by scientists at DeepMind who created a computer program called AlphaGo, which beat both a former world champion Go player and the current champion in 2016 and 2017, respectively. Many experts predicted this achievement would not come for another decade.

There are many different kinds of neural networks, but the basic notion of how they work is simple. They are loosely based on the human brain and are comprised of one or more layers of “neurons”, which are just mathematical operations that pass on a signal from the previous layer. At each layer, computations are applied on the input from neurons in the previous layer and the output is then relayed to the next layer. The output of the network’s final layer will represent some prediction about the input data, depending on the task. The challenge in building a successful neural network is finding the right computations to apply at each layer.

Neural networks can process high dimensional numerical and cateogorical data and perform tasks like regression, classification, clustering, and feature extraction. A neural network is created by first configuring its architecture based on the data and the task and then tuning its hyperparameters to optimize the performance of the neural network. Once the neural network has been trained and tuned sufficiently, it can be used to process new sets of data and return reasonably reliable predictions.

## Where Eclipse DeepLearning4j Fits In

[Eclipse Deeplearning4j](https://deeplearning4j.org/) (DL4J) is an open-source, JVM-based toolkit for building, training, and deploying neural networks. It was built to serve the Java and Scala communities and is user-friendly, stable, and well integrated with technologies such as Spark, CUDA, and cuDNN. Deeplearning4j also integrates with Python tools like [Keras](https://deeplearning4j.org/keras-supported-features) and TensorFlow to deploy their models to a production environment on the JVM. It also comes with a group of open-source libraries that Skymind bundles in an enterprise distribution called the [Skymind Intelligence Layer (SKIL)](https://skymind.ai/quickstart). Those libraries are:

* [Deeplearning4j](https://github.com/deeplearning4j/deeplearning4j/): Neural network DSL (facilitates building neural networks integrated with data pipelines and Spark) 
* [ND4J](https://github.com/deeplearning4j/nd4j/): N-dimensional arrays for Java, a tensor library: "Eclipse January with C code and wider scope". The goal is to provide tensor operations and optimized support for various hardware platforms
* [DataVec](https://github.com/deeplearning4j/datavec/): An ETL library that vectorizes and "tensorizes" data. Extract transform load with support for connecting to various data sources and outputting n-dimensional arrays via a series of data transformations 
* [libnd4j](https://github.com/deeplearning4j/libnd4j/): Pure C++ library for tensor operations, which works closely with the open-source library JavaCPP (JavaCPP was created and is maintained by a Skymind engineer, but it is not part of this project).
* [RL4J](https://github.com/deeplearning4j/rl4j/): Reinforcement learning on the JVM, integrated with Deeplearning4j. Includes Deep-Q learning used in AlphaGo and A3C.
* [Jumpy](https://github.com/deeplearning4j/jumpy/): A Python interface to the ND4J library integrating with Numpy
* [Arbiter](https://github.com/deeplearning4j/arbiter): Automatic tuning of neural networks via hyperparameter search. Hyperparameter optimization using grid search, random search and Bayesian methods. 
* [ScalNet](https://github.com/deeplearning4j/scalnet): A Scala API for Deeplearning4j, similar to Torch or Keras in look and feel. 
* [ND4S](https://github.com/deeplearning4j/nd4s/): N-Dimensional Arrays for Scala, based on ND4J.

Here are some reasons to use DeepLearning4j.

You are a data scientist in the field, or student with a Java, Scala or Python project, and you need to integrate with a JVM stack (Hadoop, Spark, Kafka, ElasticSearch, Cassandra); for example, you want to scale out your neural net training on [Spark](https://deeplearning4j.org/spark) over multi-[GPUs](https://deeplearning4j.org/gpu). You need to explore data, conduct and monitor experiments that apply various algorithms to the data and perform training on clusters to quickly obtain an accurate model for that data.

You are a data engineer or software developer in an enterprise environment who needs stable, reusable data pipelines and scalable and accurate predictions about the data. The use case here is to have data programmatically and automatically processed and analyzed to determine a designated result, using simple and understandable APIs.

# Example: Building a Feed-Forward Network

A feed forward network is the simplest form of neural networks and was also one of the first ever created. Here we will outline an example of a feed forward neural network based off an example located <a href=https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/classification/MLPClassifierMoon.java>here using moon data</a>. The data is located <a href=https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/resources/classification>here</a>.

The raw data consists of CSV files with two numerical features and two labels. The training and test sets are in different CSV files with 2000 observations in the training set and 1000 observations in the test set. The goal of the task is to predict the label given the two input features. Thus, we are interested in classification.

We first initialize the variables needed to build a feed forward neural network. We set the hyperparameters of the neural network such as the learning rate and the batch size, as well as varaibles related to its architecture, such as the number of hidden nodes.

```
int seed = 123;
double learningRate = 0.005;
int batchSize = 50;
int nEpochs = 100;

int numInputs = 2;
int numOutputs = 2;
int numHiddenNodes = 20;

final String filenameTrain  = new ClassPathResource("/classification/moon_data_train.csv").getFile().getPath();
final String filenameTest  = new ClassPathResource("/classification/moon_data_eval.csv").getFile().getPath();
```

Because the data is located in two CSV files, we initialize two `CSVRecordReaders` and two `DataSetIterators` in total. The `RecordReaders` will parse the data into record format, and the `DataSetIterator` will feed the data into the neural network in a format it can read.

```
RecordReader rr = new CSVRecordReader();
rr.initialize(new FileSplit(new File(filenameTrain)));
DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);

RecordReader rrTest = new CSVRecordReader();
rrTest.initialize(new FileSplit(new File(filenameTest)));
DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,2);
```

## Building a Feed Forward Network

Now that the data is ready, we can set up the configuration of the neural network using `MultiLayerConfiguration`.

```
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .iterations(1)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .learningRate(learningRate)
    .updater(Updater.NESTEROVS)
    .list()
    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.RELU)
        .build())
    .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.SOFTMAX)
        .nIn(numHiddenNodes).nOut(numOutputs).build())
    .pretrain(false).backprop(true).build();
```

There is one hidden layer with 20 nodes and an output layer with two nodes using a softmax activation function and a negative log likelihood loss function. We also set how the weights of the neural network are initialized and how the neural network will optimize the weights. To make the results reproducible, we also set its seed; that is, we use randomly initialized weights, but we save their random initialization in case we need to start training from the same point later, to confirm our results.

## Training and Evaluating a Feed Forward Neural Network

To actually create the model, a `MultiLayerNetwork` is initialized using the previously set configuration. We can then fit the data using a training loop; alternatively, if a `MultipleEpochsIterator` is used, then the fit function only needs to be called only once to train the data with the set amount of epochs.

```
MultiLayerNetwork model = new MultiLayerNetwork(conf);
model.init();
model.setListeners(new ScoreIterationListener(100)); 

for ( int n = 0; n < nEpochs; n++) {
    model.fit( trainIter );
}
```

Once the data has finished training, we will use the test set to evaluate our model. Note that `testIter` creates `DataSets` according to the previously set batch size of 50. The `Evaluation` class will handle computing the accuracy using the correct labels and the predictions. At the end, we can print out the results.

```
Evaluation eval = new Evaluation(numOutputs);
while(testIter.hasNext()){
    DataSet t = testIter.next();
    INDArray features = t.getFeatureMatrix();
    INDArray labels = t.getLabels();
    INDArray predicted = model.output(features,false);
    eval.eval(labels, predicted);
}

System.out.println(eval.stats());
```

This example covered the basics of using `MultiLayerNetwork` to create a simple feed-forward neural network. 

* To learn more, see our O'Reilly book: [Deep Learning: A Practitioner's Approach](http://shop.oreilly.com/product/0636920035343.do)
* And check out the [Deeplearning4j programming guide](https://deeplearning4j.org/programmingguide/01_intro)
