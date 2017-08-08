------
title: DeepLearning4j Examples Explained
layout: default 

------

# DeepLearning4j Examples Explained

Common Structure by example

- Feed Forward or MLP 
- Convolutional
- Deconstructed Example

## MultiLayerConfiguration

 conf = new NeuralNetConfiguration.Builder()

```
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

MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()

```
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
```

##  Deconstructed: Training NN code sample

[**Chris:** The balance of this file is very raw. It only has my raw notes in chunks. Don't edit this yet.]

For any dl4j-example you select, there will be areas that require editing for your specific situation. When reading the file in IntelliJ, right-click the italic objects and select from the listed choices.

The following is a annotated description of each area of a CSVExample. 

### Declarations

`package org.deeplearning4j.examples.dataexamples;`

The name of the package assigned by the creator. The package is a collection of classes that perform the tasks.

Edit to a name appropriate to your project.

`import org.datavec.api.records.reader.RecordReader;`
`...`
`import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;`
`...`
`import org.nd4j.linalg.activations.Activation;`
`...`
`import org.slf4j.Logger;`
`…`

The imports are used by the includes. 

No editing required.

Record reader > returns a list of writables. About Source content. get data > INDarray	data > writable > INDarray	see javadoc	deeplearning4j.org/datavecdoc > record reader

`public class CSVExample {`

  ` private static Logger log = LoggerFactory.getLogger(CSVExample.class);`
// Some type of logging is needed, once again not our role to explain this

  ` public static void main(String[] args) throws  Exception {`

### Defining the Conditions

//HERE IS THE BEEF

//First: get the dataset using the record reader. CSVRecordReader handles loading/parsing


       int numLinesToSkip = 0;
//SET SOME PARAMETERS
// the data file has no header
// set at the top so we can switch it easily

       String delimiter = ","; 
//FILE IS COMMA DELIMTED 
// Up to here we have actually done very little it starts soon

       RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
       recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

// MOST CODE WILL USE A RECORDREADER OF SOME SORT
// See DataVec Record Reader, also I wrote some decent docs on our main page
// I believe RecordReader returns a list of Writables
// Writables are borrowed from hadoop as an efficient serialization protocol
// Content is text or images or whatever and RecordReader parses that
// BIG PICTURE DATA ⇒ INDARRAY..
// DATA=>Writable=>INDARRAY
// DEFINE IT, then initialize by passing it a FIlePath
// ClassPAthResource is just a convenience class to make code examples portable
// Real World, >> path to file system, file object
//recordReader.initialize(new File(PATH TO FILE));

 //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network

         int labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
// File Looks like this
//5.1,3.5,1.4,0.2,0
// measurement, measurment, measurement, measurment, LABEL
// Measurements are properly features
// features => INDARRAY , Labels=> INDARRYA
// FEATURES=>NN=>emits a label
// then the output is compared to the label and then we repeat

data science use word feature for measurements, value for what measuring, label what measuring. 

label is conclusion = predict the label given the features. 

   int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2


          int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
​      

example 3 types of flowers,  

```
      int batchSize = 150;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
  
```

 batchSize — how much to pass through at a time. how often weights adjusted

// batchsize refers to how often weights are updated

       DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
// Takes writables from record reader and creates DataSet
// Dataset is INDarray of features and INDArray of Labels

standard java class that forward only, has a next method, instance of the class DataSetintrator > calling it iterator, RecordReaderDataSetintrator is a subset . inputs are parameter. Whole is called a contstructor

New instance of a sub class, using the constructor. have to match constructors defined, see javadoc for RecordReaderDataSetiterator

       DataSet allData = iterator.next();

// allData.getFeatures =>INDARRY of FEATURES
// allData.getLabels 
// Each iterator has a Next

In Intellij > allData.getFeatures= INDARRY of Features.take code, expand, intellij, allData

       allData.shuffle();
// Our batchsize is 150 and that is all our data
// Shufflle is important and you can shuffle early in the pipeline with the iterator or the file path you can shuffle as well
// Usually set random seed for reproducable results

batch size, when define iterator or filepath, can also shuffle. randomize data, read only once, random seed >> for reproducibility relative to shuffle. 


       SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  //Use 65% of data for training
// SPLIT DATA INTO TEST AND TRAIN
// At bare minimum you need two splits, you may need 3

test, train, and validate >> data does see, prevent overfitting. choice when to split the data 

       DataSet trainingData = testAndTrain.getTrain();
       DataSet testData = testAndTrain.getTest();
// You can also split at the filesplit point
       //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
       DataNormalization normalizer = new NormalizerStandardize();
       normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
       normalizer.transform(trainingData);     //Apply normalization to the training data
       normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

// Normalizing and standardizing
// Imagine you have temperature in Celsius, 
// and height in feet
// one has a range of 0-6, the other -270 to some big hot number
// so we level the unit variance 
// take max temp = 1
// take min temp =0

ex: features one w/large units, one with small units level them. so large doesn’t overwhelm by unit variance like reducing to percentage of their range. set of normalize/standardize tools common to machine learning we provide tools, check dj4j/normalizer java doc


       final int numInputs = 4;
       int outputNum = 3;
// classification outputnum= number of nodes in output layer

(input nodes,) set params for NN

4 measurement, ex. > number input nodes = number of features. typically, for this type of NN outputNum = number labels., for classification ex. Long seed — reproducible if pass seed to shuffle add to ()

       int iterations = 1000;
       long seed = 6;
/// set params for the NN
/// Number of input nodes matches number of features in FeedForwardNeuralNetwork

====

       log.info("Build model....");
       MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
           .seed(seed) // for reproducability
           .iterations(iterations) // number of updates.. BEthany see doc page on epochs batches and iterations
           .activation(Activation.TANH) \\ ACTIVATION THIS is important they will set this and it matters
// Neuron’s (output = Inputs * Weights + Bias) => Activation 
// So for a neuron here with input 5.1,3.5,1.4,0.2,0
// 5.1 * weight1 + bias1 = input 1
// 3.5 * weight2+bias2 = input 2
// 1.4 * weight3 + bias3 = input 3

(Input1 + ,2,+ 3,+ 4) => ACTIVATION => OUTPUT TANH is a value beetween -1 and +1

INput Iris label 0 => random label 
See SImplest network example from the training class_ 

builder method, standard for NN, but  MultiLayerConfiguration >> diff ways to do this, we are calling NeualaNEtConfiguaraiton.Builder

           .weightInit(WeightInit.XAVIER)
// randomizes initial weights
           .learningRate(0.1)
// Important how big of steps it takes to correct error
           .regularization(true).l2(1e-4) // ask me later
// Using what is called a builder method
// way to deal with complex variable sized inputs
// the .list might be important

way to handle complex variable size, flexible with number

           .list()

Below is where we build it
           .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
               .build())
//Input Layer 4 nodes nEYeN
           .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
               .build())
// RUN with  UI attached to see image of the layers
// 4=>3=>3=>3

Section users/dev add. iterations > can also use epoch, batches, iterations

           .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
               .activation(Activation.SOFTMAX)
               .nIn(3).nOut(outputNum).build())
           .backprop(true).pretrain(false)
           .build();
    
       //run the model
       MultiLayerNetwork model = new MultiLayerNetwork(conf);
       model.init();
       model.setListeners(new ScoreIterationListener(100));
    
       model.fit(trainingData);
    
       //evaluate the model on the test set
       Evaluation eval = new Evaluation(3); // The Evaluation class is determined by what is imported
### Evalution

// SO EVALUATION is actually this. import org.deeplearning4j.eval.Evaluation;
// So that makes it org.deeplearning4j.eval.Evaluation 

Activation > determines final output, formulas, one weight per each.  > send to activation function. ex. 5.1, 3.5, 1.4 normalized, sample formulas each input formulas

Neuron takes inputs, add them, applies activation function > produces output of activation function. 

ex output TANH is a value between -1 and +1, min and max of output. step function increases in middle, keep values in reasonable range. weights updated as NN learns

***Take inputs, multiple by random weights and biases, pass through activation > output compare output to expected output. adjust weights and biases. rerun

***

usually batch is smaller, number of passes is by epoch. ex is not standard bc small. weighting >> Xavier, almost always good. randomizes weights. learningRate > step size, 

====

build NN >> layers. nin, Number of inputs, 4 inut, 3 hidden. 1st layer numInputs is 4, output > 2 layers. second layers each 4, then 3, then 3. 

       INDArray output = model.output(testData.getFeatureMatrix());
       eval.eval(testData.getLabels(), output);
       log.info(eval.stats());
       log.info(eval.confusionToString());
       log.info(String.valueOf(eval.falsePositiveRate(0)));
    
       //RegressionEvaluation regressionEvaluation = new RegressionEvaluation()
   }

}

Here is the same example in Python…

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



