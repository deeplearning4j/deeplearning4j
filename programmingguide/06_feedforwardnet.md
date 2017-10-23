---
title: DeepLearning4j Feed Forward Network
layout: default
---

# DeepLearning4j: Feed-Forward Network Example

A feed forward network is the simplest form of neural networks and was also one of the first ever created. Here we will outline an example of a feed forward neural network based off an example located [here](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/classification/MLPClassifierMoon.java) using moon data. The data is located [here](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/resources/classification). 

- [**Data & ETL**](#ETL) 
- [**Building a LSTM Network**](#Building) 
- [**Training & Evaluating**](#Training) 

## <a name="ETL">Data and ETL</a>

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

## <a name="Building">Building a Feed Forward Network</a>

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

## <a name="Training">Training and Evaluating a Feed Forward Neural Network</a>

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

This example covered the basics of using `MultiLayerNetwork` to create a simple feed-forward neural network. Stay tuned for the following chapter, which will cover more advanced uses of DL4J like Natural-Language Processing (NLP).
