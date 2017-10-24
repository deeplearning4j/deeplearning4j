---
title: DeepLearning4j LSTM Network
layout: default
---

# DeepLearning4j: LSTM Network Example

In this section, we will cover an example of an LSTM (long short term memory) neural network. An LSTM network is a type of recurrent network where the input is usually sequential in nature, such as speech, text or time series. 

At each timestep of the data, the neural network ingests input features and the output is then passed onto the next timestep. If the output is also time series in nature, then the neural network saves the output at each timestep (e.g. translation). If not, then only the output of the last timestep is taken (e.g. classifying a sequence). 

Thus, we can see that recurrent neural networks differ from a regular feed forward neural network (FNN), since FNNs only take in one set of features per observation, and produce one output. They are one to one. Recurrent neural networks are not bound to this constraint.

An LSTM is a special type of recurrent network using a gated "cell" where the network can store and remove information. 

The gate of the cell filters the content to keep only relevant information. At each timestep, the neural network takes in both new input features and the content from the gated cell from the previous time step. LSTM's are useful because they mitigate vanishing or exploding gradients that hinder a neural network from learning. For a more in depth overview of LSTMs, look [here](https://deeplearning4j.org/lstm.html).

The following phases are included:

- [**Data and ETL**](#ETL) 
- [**Building a LSTM Network**](#Building) 
- [**Training and Evaluating**](#Training) 

## <a name="ETL">Data and ETL</a>

The data used for this example is from the [2012 Physionet Challenge](https://physionet.org/challenge/2012/). It consists of Intensive Care Unit (ICU) visits by 4000 patients, and it is comprised of 86 features and a binary mortality label. 

The number of timesteps varies between each patient, and the time elapsed between timesteps vary as well. The features of an observation are stored in their own CSV file with the label in a separate CSV file in a different directory. All the feature files are contained in the same directory and the same goes for the label files. Additionally, the feature file and label file of the first observation is named 0.csv, while the feature file and label file of the second observation is named 1.csv and etc.

To start the process of extracting, transforming and loading data (ETL), we use a `CSVSequenceRecordReader` to parse the data into a vectorized format. We take the first 3,200 observations as the training portion of the data. 
		
```
private static File featuresDir = new File(path); // path to directory containing feature files
public static final int NB_TRAIN_EXAMPLES = 3200;

SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1, ",");  // number of rows to skip + delimiter
trainFeatures.initialize( new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));
SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
trainLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));
```
A new method in the above code is NumberedFileInputSplit. The NumberedFileInputSplit takes in the path + file name, which is in the form of path/%d.csv where %d stands for some integer value for this case. Because we specify the first 3,200 observations as the training split, files from 0.csv to 3199.csv are included in the split.

Once we have the `RecordReaders`, we need to initialize the `DataSetIterator`. 

```
int numLabelClasses = 2;

DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels,
    BATCH_SIZE, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
```

The DataSetIterator for SequenceRecordReaders differs slightly from those used for CNN's or FNN's. It requires an alignment mode, which aligns the input and label of each observation. In this example, ALIGN_END is most appropriate. We will use this specific alignment mode, because of the varying number of timesteps between observations and the difference in the number of inputs and outputs for each observation. The former difference is due to the fact that the patients in the dataset have differing numbers of measurements during the ICU stay. The latter difference is because the mortality output is taken only at the last timestep while the inputs have multiple timesteps.

The way ALIGN_END works is that it pads (add zero values to) the time series observations so all observations have the same number of timesteps. Thus, all  observations will have the same number of timesteps as the observation with the greatest number of timesteps. However, having these added zero values in the input and output arrays will affect the training process. Thus, two additonal masking arrays are used to indicate whether an input or output is actually present at a given timestep or whether they are just padding. This also fixes the issue of the differing number of inputs and outputs, since the outputs are padded to match the inputs. To learn about this masking process in greater depth, look [here](https://deeplearning4j.org/usingrnns#masking).

## <a name="Building">Building a LSTM Neural Network</a>

Now that the training data is in a format that a neural network can read, we can start building the neural network. Unlike previous examples, we will use the `ComputationGraph` class instead of `MultiLayerNetwork` although `MultiLayerNetwork` can be used instead.

```
public static final double LEARNING_RATE = 0.05;
public static final int lstmLayerSize = 300;
public static final int NB_INPUTS = 86;

ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .learningRate(LEARNING_RATE)
    .graphBuilder()
    .addInputs("trainFeatures")
    .setOutputs("predictMortality")
    .addLayer("L1", new GravesLSTM.Builder()
        .nIn(NB_INPUTS)
        .nOut(lstmLayerSize)
        .activation(Activation.SOFTSIGN)
        .weightInit(WeightInit.DISTRIBUTION)
        .build(), "trainFeatures")
    .addLayer("predictMortality", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .activation(Activation.SOFTMAX)
        .weightInit(WeightInit.DISTRIBUTION)
        .nIn(lstmLayerSize).nOut(numLabelClasses).build(),"L1")
    .pretrain(false).backprop(true)
    .build();
```
We can see that the neural network code for building a `ComputationGraph` is similar to that of `MultiLayerNetwork`. To build the LSTM layer, the `GravesLSTM.Builder()` class is used with the specified dimension of a timestep and layer size. Furthermore, the output layer has a output dimension of 2, since we are classifying a binary mortality label using the data. 

Next, initialize the actual `ComputationGraph` using the previously specified configuration.

```
ComputationGraph model = new ComputationGraph(conf);
model.init();
```

## <a name="Training">Training and Evaluating a LSTM Neural Network</a>

The code to train an LSTM network doesn't change from previous examples. The `ComputationGraph` class takes care of the training process and all that's needed is the training loop for the number of epochs. Here we train the neural network using 15 epochs, or "passes", through the training set.

```
public static final int NB_EPOCHS = 15;

for( int i=0; i<NB_EPOCHS; i++ ){
    model.fit(trainData); 
}
```

To evaluate the trained neural network, we compute the AUC (area under the curve) for a Receiver Operating Chracteristic (ROC) curve with the output predictions from the neural network and the labels of the observations. Note that a perfect model will have an AUC of 1.0 and a randomly guessing model will achieve a AUC near 0.5. 

Although we have not shown the code to process the testing split of the data, we assume it is contained in the DataSetIterator called testData. The ETL process for the test split is similar to the process shown above for the training split.

```
ROC roc = new ROC();
while (testData.hasNext()) {
    DataSet batch = testData.next();
    INDArray[] output = model.output(batch.getFeatures());
    roc.evalTimeSeries(batch.getLabels(), output[0]);
}
log.info("FINAL TEST AUC: " + roc.calculateAUC());
```

### DL4J's Programming Guide  

* [1. Intro: Deep Learning, Defined](01_intro)
* [2. Process Overview](02_process)
* [3. Program & Code Structure](03_code_structure)
* [4. Convolutional Network Example](04_convnet)
* [5. LSTM Network Example](05_lstm)
* [6. Feed-Forward Network Example](06_feedforwardnet)
* [7. Natural Language Processing](07_nlp)
* [8. AI Model Deployment](08_deploy)
* [9. Troubleshooting Neural Networks](09_troubleshooting)
