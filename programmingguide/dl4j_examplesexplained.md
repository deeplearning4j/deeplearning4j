------
title: DeepLearning4j Examples Explained
layout: default 

------

# DeepLearning4j Examples Explained

The example files are located at: https://github.com/deeplearning4j/dl4j-examples

Note that the DeepLearning4J project uses and recommends Maven for dependency management. Maven uses `pom.xml` files for configuration. Note that the examples make use of a two-level `pom.xml` structure, one `pom.xml` file in the main directory and another one in the base directory. 

Files that require editing for each specific training scenario. 

* pom.xml
  Purpose
  Required content - dependencies
  What to edit - dependencies
  Options (to select, edit, or add)

Specific examples of different neural network layers are here:
https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples

For most common neural net applications, you will find an example that corresponds to your use case.

* Feed Forward Neural Networks for Classification
* Recurrent Neural Networks for working with Time Series Data
* Convolutional Neural Networks for dealing with images

Locate the file that corresponds to the type of training you want. Review the content to define the particulars as listed below.

Common Structure by example

- Feed Forward or MLP 
- Convolutional
- Deconstructed Example

## MultiLayerConfiguration

Extracted portion of an example for building the neural network layer in a dense multi-layer configuration. 

Here is an example of a Feed Forward Neural Network Configuration also known as a Multi Layer Perceptron. A Feed Forward Neural Network consists of an Input Layer, an Output Layer and a user determined number of Fully Connected or Dense Layers. 

```
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
```

## Convolutional

### Convolutional Input
A Convolutional Neural Network takes an input matrix that includes the depth of the input, for example an image has a value for each pixel to represent amount of green, amount of red, and amount of blue. A Convolutional Neural Networks input includes that structure. 

[See](https://deeplearning4j.org/convolutionalnets.html)

### Convolutional Layers
A Convolutional Layer 

[See](https://deeplearning4j.org/convolutionalnets.html#work)

Typical Pattern in a Convolutional Network is a Convolutional Layer followed by a SubSampling, typically a maxPooling, layer. The Convolutional Layer extracts small scale patterns, the SubSampling Layer summarizes or reduces the number of Parameters introduced by the convolutional Layer. 

Extracted portion of an example for building the neural network layer in a convolutional multi-layer configuration. 

```
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
   .seed(rngseed)
   .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
   .iterations(1)
   .learningRate(0.006)
   .updater(Updater.NESTEROVS).momentum(0.9)
   .regularization(true).l2(1e-4)
   .list()
     .layer(0, new ConvolutionLayer.Builder(5, 5)
            .nIn(channels)
            .stride(1,1)
            .nOut(20)
            .activation(Activation.IDENTITY)
            .build())
       //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
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
```

##  Deconstructed: Training neural network code sample

A first step would be to find an example that is similar to your needs and use it as a template to build your own network. 
Changes that will be required. 
* Your Input Layer
* Your Output Layer

The structure of your Input Layers is determined by the format of your input. Take a look at the CSV example to clarify this point. The dataset has 4 features, sepal length, sepal width, petal length and petal width. This correlates with the size or number of nodes in the input layer; that is, four features mean four numbers are fed to the net. 

The structure of your Output Layer is determined by the task you seek to perform with the Neural Network. When asked to perform classification, the output layer will have one neuron for each class or category you want to apply to the data (e.g. for face identification on images, you'd have one output node per name). The CSV example illustrates this: We have flowers of three classes, *iris setosa*, *iris virginica*, and *iris versicolor*. We therefore have three neurons in the output Layer. 

If you were to model your neural net for a similar role but had 5 features and 2 classes, instead of 4 features and 3 classes in the CSV example, you could take the code in the CSV example and simply change the Input Layer and Output Layer. 

For any dl4j-example you select, there will be code and parameters that require editing to adapt them to your specific task and data. When reading the file in IntelliJ, right-click the italic objects and select from the listed choices.

The following is an annotated description of each area of a `CSVExample`. You edit this file through IntelliJ, then use Maven, with the edited `pom.xml` file to build the neural network model. 

### Declarations

The top portion of the file, typical to most programming files, contains the declarations, parameters, and identification information. Using the example files provided, some of the elements remain as is, others you edit to your specifics. 

**Package name**

The name of the package assigned by the creator. The package is a collection of classes that perform the tasks. Edit to a name appropriate to your project.

```
package org.deeplearning4j.examples.dataexamples;
```

**Imports**

The imports are used by the includes. The list is specific to the use case/example. No editing required. Click the `...` to expand and view the list of imports.

```
import org.datavec.api.records.reader.RecordReader;
...
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
...
import org.nd4j.linalg.activations.Activation;
...
import org.slf4j.Logger;
…
```

## Reading Data

DeepLearning4J uses `RecordReaders` to help take the data from raw source to an n-dimensional array of numeric values, or `INDArray`, the format ingested by a Neural Network. A `RecordReader` is the first step in that process. Different RecordReaders are available for processing images, audio, text and time series data. 

*  RecordReader

   See the Javadoc at https://deeplearning4j.org/datavecdoc/ > recordreader. 

   * RecordReader -- is in all examples. It returns a list of writables. Converting the source content into numeric values. 
   * RecordReaderDataSetIterator -- is in all examples. It converts the numeric values into n-dimentional arrays (n-d arrays) approproriate to the use case. 

* Logging

   * Logger -- is in all examples. Some form of logging is required. 

**Neural Network Class Type**

Define the class type for the neural network and some file error handling.

```
public class CSVExample {
   private static Logger log=LoggerFactory.getLogger(CSVExample.class);
   public static void main(String[] args) throws  Exception {
```

### Defining the Conditions

The edits in this section describe the conditions you are applying to the neural network training. 

#### Step 1

**Retrieve the dataset**

Use the RecordReader imported above. 

```
int numLinesToSkip = 0;
       String delimiter = ",";
```

In this example, the CSV (comma-separated values) RecordReader handles loading and parsing the data. 

There is no header in the csv file, so the first entry of data is at line 0. Specify 0 lines to skip in the file. The file delimiter, used to separate data is a comma.

**Instantiate the RecordReader class**

```
       RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
       recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
```

Most neural network models require a RecordReader. The RecordReader imports and converts the raw data into writables. The writeables are passed through to the neural network in n-dimensional (N-D) arrays. These writables, borrowing from Hadoop, use an efficient serialized protocol to process source content that is text, images, or other form of data.

Define the RecordReader, initialize it by passing it a file path. For example, ClassPathResource is a convenient class to make code examples portable, it follows the format: 

`recordReader.initialize(new File(new PATH TO FILE))`

Python user tip: Common Java format: instantiate the class, assign it a name, declare it as new, then initialize it with listed parameters. 

#### Step 2

Define how the DataSetIterator handles conversion of the writables into DataSet objects that are ready for use in the neural network. 

**Specify the number of labels, classes, and batch size**

```
  int labelIndex = 4;
   int numClasses = 3;
   int batchSize = 150;
```

This supports the process: feed in the features into the neural network, which generates an output, a prediction about the label. The output is compared to the label. The process is repeated until the output within an acceptable range, agrees with the label.

For example, a row of data in the source file that contains measurements of Iris flowers. 

  `5.1,3.5,1.4,0.2,0`

* Classes are the features that are measured. These are the source data values. 

   The first 4 values are measurements of a flower. These are sepal length, sepal width, petal length, and petal width.

* Labels are the target answers we are asking the neural network to identify. In this example, the goal of training the neural network is to accurately predict the label.

   The last value is the index label for the row. The label options here are 0, 1, or 2, representing the 3 species of Irises measured.

* Batch size specifies how much data to pass through at a time and how often the weights are adjusted. Weights are adjusted each time a batch of data is processed. The batch size in the Iris example, the entire dataset is entered in one batch. This is not recommended for large datasets. 

**Instantiate the DataSetIterator class**

See javadoc for RecordReaderDataSetIterator

```
    DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses); 
```

The DataSetIterator takes the writables from the RecordReader and creates the DataSet. The DataSet is two arrays, an N-D array of features and an N-D array of labels.

Python user tip: Standard Java classes that forward only have a next method. For example, an instance of the class, DataSetIterator, called `iterator`, creates a subset RecordReaderDataSetIterator. The inputs are the parameters. The entire string is called a constructor. When you issue a new instance of the subclass, use the constructor. Ensure that the defined constructors match. 

**Set the DataSet class**

```DataSet allData = iterator.next();```

Every iterator requires a next. 

In this example, all the data is imported and used in each pass. 

In other situations, the data can be split into identified data. For example, allData.getFeatures and allData.getLabels. Each of these pull in an NDarray of features and an NDarray of labels. 

**Shuffling the Data**

Training progresses smoothly if the data has been shuffled. There are a number of ways to shuffle a dataset, in this case our data is small enough to fit as dataset in memory, so we can use the shuffle method of the dataset class. 

```allData.shuffle();```

Set the data that is to be shuffled in each pass. The shuffle can be set when either the iterator or the filepath is defined. 

In this example, the batch size is 150, which is all of the data in the sample. All the data, the full batch size is included in each shuffle.

A batch size is defined with either the iterator or the filepath. It can also be shuffled. This randomizes the data prior to reading it based on a random seed. Batch size and a random seed are set for reproducible results relative to the shuffle. 

**Split the data into a test set and a training set**

```
   SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
   DataSet trainingData = testAndTrain.getTrain();
   DataSet testData = testAndTrain.getTest();
```

Splitting the data, sets a portion of the data for each function, training and testing. This can be used to detect overfitting, when a network performs well on data it has trained on, but performs poorly on unseen data. Minimum required is two splits, for training the model and testing the model. It is common to have three splits: one each for training, testing, and validating. Optionally, the split allocation can be set at the file split point instead.

Define the data sets for the training data and testing data. In this example, 65% of the data is set for training.

**Normalize the data**

```DataNormalization normalizer = new NormalizerStandardize();
       normalizer.fit(trainingData);           
       normalizer.transform(trainingData);     
       normalizer.transform(testData);
```

The raw data needs to be normalized to level the unit variance between features. This is needed to ensure that a feature with large units does not overwhelm the other features.  For example if one features measures adult height, with a range of 21" to 99", and another feature measures populations of countries, with a range of (450 to 1,388,232,693), normalizing adjusts the values to fit within the same range. This example uses NormalizeStandardize, this returns mean 0 and unit variance. See the javadoc, https://deeplearning4j.org/datavecdoc/ > Normalization.

Instantiate the DataNormalization. Then set the data purposes. In order, the `normalizer.`*action* lines perform: 

* Collect the statistics (mean/stdev) from the training data. This does not modify the input data
* Apply normalization to the training data
* Apply normalization to the test data. This is using statistics calculated from the training set

**Set the layer and iteration parameters**

```
  final int numInputs = 4;
  int outputNum = 3;
  int iterations = 1000;
  long seed = 6;
```

* Input and output nodes. 
  For the layer, set the number of input nodes to the layer and the number of output nodes from the layer. Typically, for a classification model, the number of input nodes are the number of features and the number of output is the number of labels. 

* Iterations.
  The number of times the passes and updates made with the data. See also epochs and batches.

[epoch](https://deeplearning4j.org/glossary#epoch)
[iteration](https://deeplearning4j.org/glossary#iteration)

* Seed.
  This is used with each batch and shuffle to ensure reproducibility. It sets a starting point for randomizing the data. In this example, the value is set to 6. Rather than set a specific number, another common option is using a random number generator (rng). Example of randomizing a FileSplit using a predefined Random **from another example**. 
  
  ```
   int rngseed = 123;
   Random randNumGen = new Random(rngseed);
   FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

  ```
This completes the portion that defines the input and output options.

#### Step 3

Define how the neural network model proceeds.

**Instantiate the model class**

This is performed referencing the objects defined above. 

```
   log.info("Build model....");
   MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
     .seed(seed)
     .iterations(iterations)
```

* Reproducibility object. Using seed. Defined above, applied here.
* Number of iterations. Number of updates applied to the data. Defined above, applied here.

**Activation**

```
    .activation(Activation.TANH)
```

Type of activation. Using TANH. Select from options. 

Activation type sets how the output values are evaluated.  The neuron's output, calculated from inputs * weights + bias, is fed into the activation. 

For example. A neuron with inputs: 5.1,3.5,1.4,0.2,0
5.1 * weight1 + bias1 = input 1
3.5 * weight2+bias2 = input 2
1.4 * weight3 + bias3 = input 3

The input values, as processed, are fed to through the activation. 
The output of a TANH activation is a value between -1 and +1. 

Initial pass, input labels the Iris = 0, a random label. Each pass evaluation of likelihood this is correct label, adjusts the weights and bias. 

Other methods for process call the NeuralNetConfiguration.Builder.

**Randomization method**

```
    .weightInit(WeightInit.XAVIER) 
```

This example uses the XAVIER method for randomizing the initial weights. Xavier is a good randomized weight initialization algorithm that takes into account the number of connections a neuron has when choosing initial weights.  

**Step size for each iteration**

```
   .learningRate(0.1)
```

Set the size of the steps to use to correct the error. Too large and you overshoot, too small and it takes longer to get to an appropriate solution.

**Set regularization formula **

```
    .regularization(true).l2(1e-4)
    .list()
```

* Regularization is integrated with the builder method applied. Regularization helps by decreasing variance among the weights, L1 regularization favors a distribution with many zeros while L2 penalizes large weights. Both can help a neural network train more efficiently and avoid overfitting. 

[Good discussion here on quora.] 
(https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization-How-does-it-solve-the-problem-of-overfitting-Which-regularizer-to-use-and-when)

* If you are unfamiliar with using a builder pattern for creating complex objects see the [wikipedia page](https://en.wikipedia.org/wiki/Builder_pattern)

#### Step 4
Build the neural network model

**Define input  and output layers**

```
  .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
    .build())
  .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
    .build())
  .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
	.activation(Activation.SOFTMAX)
	.nIn(3).nOut(outputNum).build())
```

* Input Layer is determined by data. How many features types are input. This is layer 0.
* Hidden Layer is determined by the methods you choose to apply. At each iteration, batch, or epoch the weights and biases are adjusted through each hidden layer. In this example, there is only one hidden layer, Layer 1. 
* Output Layer is determined by question we are asking. How many options for solution. This is layer 2. 

For example, with the Iris data, there are 4 features measured -- the inputs to the first layer -- and 3 types of Irises to identify -- the output from the output layer. 

**Set backpropagation, pretraining, and build**

```
 .backprop(true).pretrain(false)
 .build();
```

Pretraining applies to unsupervised training only. For example, Variational AutoEncoders and Restricted Boltzman machines.


#### Step 5
**Run the model**

```MultiLayerNetwork model = new MultiLayerNetwork(conf);
       model.init();
       model.setListeners(new ScoreIterationListener(100));
       model.fit(trainingData);
```

Instantiating the model calls all the subset and defined parameters set above. A score is an output of loss function, and the loss function is a determiniation of error. As the passes are made, the score should move towards 0.

Example: ouput of a score iteration listener

```
o.d.o.l.ScoreIterationListener - Score at iteration 0 is 1.3087471277551905
o.d.o.l.ScoreIterationListener - Score at iteration 100 is 0.3386059108000891
o.d.o.l.ScoreIterationListener - Score at iteration 200 is 0.15180469491523504
o.d.o.l.ScoreIterationListener - Score at iteration 300 is 0.08593290147904636
o.d.o.l.ScoreIterationListener - Score at iteration 400 is 0.06059342302219454
o.d.o.l.ScoreIterationListener - Score at iteration 500 is 0.046881126863677
```

#### Step 6
**Evaluate the model results**

```
Evaluation eval = new Evaluation(3);
```

The Evaluation class is determined by what is imported. In this example, the evaluation is:

```
import org.deeplearning4j.eval.Evaluation;
```

Summary:

Activation determines final output, formulas, one weight per each.  Send to activation function the normalized data. Normalized from, for example:  5.1, 3.5, 1.4. Add sample formulas to each input formulas. 

Neuron takes inputs, add them, applies the activation function that produces output of activation function. For example, output TANH is a value between -1 and +1, min and max of output. The step function increases in the middle/hidden layers, keeps values in reasonable range. Weights updated as the neural network learns. 

Neural network takes inputs, multiple by random weights and biases, pass through activation to produce output, then compares output to expected output and adjusts weights and biases. Repeat. 

Typically, batch is smaller, number of passes is by epoch. The examples used here are not standard because they are small. 

For weighting, Xavier, is almost always good. It randomizes weights. LearningRate is set by step size.

**Build the neural network**

```
   INDArray output = model.output(testData.getFeatureMatrix());
   eval.eval(testData.getLabels(), output);
   log.info(eval.stats());
   log.info(eval.confusionToString());
   log.info(String.valueOf(eval.falsePositiveRate(0)));
```

Run the model. Run the evaluation. Create and update the logs. 
RegressionEvaluation regressionEvaluation = new RegressionEvaluation()


## Python Example

```
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

\# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

\# load dataset
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
print(X)
print(Y)

\# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


\# convert integers to dummy variables (hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y)

\# define baseline model
\#def baseline_model():
\# create model
model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dense(3,activation='sigmoid'))

\# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

\#    return model
\#model.fit
model.fit(X, dummy_y, nb_epoch=200, batch_size=5)
prediction = model.predict(numpy.array([[4.6,3.6,1.0,0.2]]));
print(prediction);


\# To casve just the weights
model.save_weights('/tmp/iris_model_weights')

\# To save the weights and the config
\# Note this is what is used for this demo
model.save('/tmp/full_iris_model')

\# To save the Json config to a file
json_string = model.to_json()
text_file = open("/tmp/iris_model_json", "w")
text_file.write(json_string)
text_file.close()
```
