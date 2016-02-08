---
title: Deep-Belief Network Tutorial for Iris
layout: default
---

# Deep-Belief Network Tutorial for Iris

**The output of a neural net training on Iris may vary, because it is a small data set.**

Deep-belief networks are multi-class classifiers. Given many inputs belonging to various classes, a DBN can first learn from a small training set, and then classify unlabeled data according to those various classes. It can take in one input and decide which label should be applied to its data record. 

Given an input record, the DBN will choose a label from a set of labels. This goes beyond a Boolean ‘yes’ or ‘no’ to handle a broader, multinomial taxonomy of inputs. 

The network outputs a vector containing one number per output node. The number of output nodes equals the number of labels. Each of those outputs are going to be a 0 or 1, and taken together, those 0s and 1s form the vector. 

![image of nn multiclassifier here](http://i.imgur.com/qfQWwHB.png)

### The IRIS Dataset

The [Iris flower dataset](https://archive.ics.uci.edu/ml/datasets/Iris) is widely used in machine learning to test classification techniques. We will use it to verify the effectiveness of our neural nets.

The dataset consists of four measurements taken from 50 examples of each of three species of Iris, so 150 flowers and 600 data points in all. The iris species have petals and sepals (the green, leaflike sheaths at the base of petals) of different lengths, measured in centimeters. The length and width of both sepals and petals were taken for the species Iris setosa, Iris virginica and Iris versicolor. Each species name serves as a label. 

The continuous nature of those measurements make the Iris dataset a perfect test for continuous deep-belief networks. Those four features alone can be sufficient to classify the three species accurately. That is, Success is teaching a neural net to classify by species the data records of individual flowers while knowing only their dimensions, and failure to do so is a very strong signal that your neural net needs fixing. 

The dataset is small, which can present its own problems, and the species I. virginica and I. versicolor are so similar that they partially overlap. 

Here is a single record:

![data record table](../img/data_record.png)

While the table above is human readable, Deeplearning4j’s algorithms need it to be something more like

     5.1,3.5,1.4,0.2,i.setosa

In fact, let’s take it one step further and get rid of the words, arranging numerical data in two objects:

Data:  5.1,3.5,1.4,0.2    
Label: 0,1,0

Given three output nodes making binary decisions, we can label the three iris species as 1,0,0, or 0,1,0, or 0,0,1. 

### Loading the data

DL4J uses an object called a DataSet to load data into a neural network. A DataSet is an easy way to store data (and its associated labels) which we want to make predictions about. The columns First and Second, below, are both NDArrays. One of the NDArrays will hold the data’s attributes; the other holds the label. (To run the example, [use this file](https://github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/iris/IrisExample.java).)

![input output table](../img/ribbit_table.png)

(Contained within a DataSet object are two NDArrays, a fundamental object that DL4J uses for numeric computation. N-dimensional arrays are scalable, multi-dimensional arrays suitable for sophisticated mathematical operations and frequently used in scientific computing.) 

Most programmers are familiar with datasets contained in file like CSVs (comma-separated value), and the IRIS dataset is no exception. Here's how you parse an Iris CSV and put it in the objects DL4J can understand. 

    File f = new File(“Iris.dat”);
    InputStream fis = new FileInputStream(f);

    List<String> lines = org.apache.commons.io.IOUtils.readLines(fis);
    INDArray data = Nd4j.ones(to, 4);
    List<String> outcomeTypes = new ArrayList<>();
    double[][] outcomes = new double[lines.size()][3];

Let’s break this down: iris.dat is a CSV file containing the data we need to feed the network.

Here, we use *IOUtils*, an Apache library, to help read the data from a file stream. Please note that readLines will copy all the data into memory (generally you shouldn’t do that in production). Instead, consider a BufferedReader object.

The NDArray variable *data* will hold our raw numeric data and the list *outcomeTypes* will be a sort of map that contains our labels. The Dataset object *completedData*, which at the end of the code you'll see below, contains all of our data, including binarized labels. 

The variable *outcomes* will be a two-dimensional array of doubles that has as many rows as we have records (i.e. lines in iris.dat), and as many columns as we have labels (i.e. the three species of iris). This will contain our binarized labels.

Take a look at this code segment

    for(int i = from; i < to; i++) {
        String line = lines.get(i);
        String[] split = line.split(",");

         // turn the 4 numeric values into doubles and add them
        double[] vector = new double[4];
        for(int i = 0; i < 4; i++)
             vector[i] = Double.parseDouble(line[i]);

        data.putRow(row,Nd4j.create(vector));

        String outcome = split[split.length - 1];
        if(!outcomeTypes.contains(outcome))
            outcomeTypes.add(outcome);

        double[] rowOutcome = new double[3];
        rowOutcome[outcomeTypes.indexOf(outcome)] = 1;
        outcomes[i] = rowOutcome;
    }

    DataSet completedData = new DataSet(data, Nd4j.create(outcomes));

O.K., time to look at  what we wrote.

Line 3: Since we’re dealing with CSV data, we can just use *split* to tokenize on each comma and store the data in the String array *split*.

Lines 6 - 10: Our String objects are strings of numbers. That is, instead of a double of 1.5, we have a String object with the characters “1.5”. We’ll create a temporary array called *vector* and store the characters there for use later. 

Line 12-14: We get the labels by grabbing the last element of our String array. Now we can think about binarizing that label. To do that, we’ll collect all the labels in the list outcomeTypes, which is our bridge to the next step.

Lines 16-18: We start to binarize the labels with our outcomeTypes list. Each label has a certain position, or index, and we’ll use that index number to map onto the label row we make here. So, if *i. setosa* is the label, we’ll put it at the end of the outcomeTypes list. From there, we’ll create a new label row, three elements in size, and mark the corresponding position in rowOutcome as the 1, and 0 for the two others. Finally, we save rowOutcome into the 2D array outcomes that we made earlier. 

By the time we finish, we'll a row with a numeric representation of labels. A data record classified as *i. setosa* would look like:

![final table](../img/final_table.png)

The words you see in the upper boxes only serve to illustrate our tutorial and remind us which words go with which numbers. The numbers in the lower boxes will be what appear as a vector for data processing. In fact, the bottom, finished row is what we call *vectorized data*.

Line 21: Now we can start to think about packaging the data for DL4J. To do that, we create a single *DataSet* object with the data we want to work with and the accompanying, binarized labels.

Finally, we’ll return the list completedData, a dataset our deep-belief network can work with. 

### Creating a Neural Network (NN)

Now we’re ready to create a deep-belief network, or DBN, to classify our inputs.

With DL4J, creating a neural network of any kind involves several steps. 

First, we need to create a configuration object:

    NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
    .hiddenUnit(RBM.HiddenUnit.RECTIFIED).momentum(5e-1f)
        .visibleUnit(RBM.VisibleUnit.GAUSSIAN).regularization(true)
        .regularizationCoefficient(2e-4f).dist(Distributions.uniform(gen))
        .activationFunction(Activations.tanh()).iterations(10000)
        .weightInit(WeightInit.DISTRIBUTION)
    .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen)
        .learningRate(1e-3f).nIn(4).nOut(3).build();

This has everything that our DBN classifier will need. As you can see, there are a lot of parameters, or ‘knobs’, that you will learn to adjust over time to improve your nets’ performance. These are the pedals, clutch and steering wheel attached to DL4J's deep-learning engine. 

These include but are not limited to: the amount of momentum, regularization (yes or no) and its coefficient, the number of iterations, the velocity of the learning rate, the number of output nodes, and the transforms attached to each node layer (such as Gaussian or Rectified). 

We also need a random number generator object:

        RandomGenerator gen = new MersenneTwister(123);

Finally, we create the DBN itself: dbn

    DBN dbn = new DBN.Builder().configure(conf)
        .hiddenLayerSizes(new int[]{3})
        .build();
      dbn.getOutputLayer().conf().setActivationFunction(Activations.softMaxRows());
      dbn.getOutputLayer().conf().setLossFunction(LossFunctions.LossFunction.MCXENT);

Let’s analyze the code above. On the first line, we take the Configuration object we called ‘conf’ and we pass it in as a parameter. Then we specify the hidden layer size. We can do so with a separate array for each layer of the array. In this case, there is a single hidden layer, three nodes long. 

Now we can prepare the DataSet object we made earlier. We’ll put it in a separate function irisLoad().

    DataSet ourDataSet = loadIris(0, 150);
    ourDataSet.normalizeZeroMeanZeroUnitVariance();
    dbn.fit(ourDataSet);

Note the second line above. In many machine-learning models, it’s important to normalize the data to ensure that outliers don’t distort the model. (The normalization of numbers means adjusting values that may be measured on different scales (in tens or hundreds or millions) to a notionally common scale, say, between 0 and 1. You can only compare apples to apples if everything is apple-scale...

Finally, we call *fit* to train the model on the data set. 

By training a model on a dataset, your algorithm learns to extract those specific features of the data that are useful signals for classifying the target input, the features that distinguish one species from another.

Training is the repeated attempt to classify inputs based on various machine-extracted features. It consists of two main steps: the comparison of those guesses with the real answers in a test set; and the reward or punishment of the net as it moves closer to or farther from the correct answers. With sufficient data and training, this net could be unleashed to classify unsupervised iris data with a fairly precise expectation of its future accuracy. 

You should see some output from running that last line, if debugs are turned on. 

### **Evaluating our results**

Consider the code snippet below, which would come after our *fit()* call.

    Evaluation eval = new Evaluation();
    INDArray output = d.output(next.getFeatureMatrix());
    eval.eval(next.getLabels(),output);
    System.out.printf("Score: %s\n", eval.stats());
    log.info("Score " + eval.stats());

DL4J uses an **Evaluation** object that collects statistics about the model’s performance. The INDArray output is created by a chained call of *DataSet.getFeatureMatrix()* and **output**. The getFeatureMatrix call returns an NDArray of all data inputs, which is fed into **output()**. This method will label the probabilities of an input, in this case our feature matrix. *eval* itself just collects misses and hits of predicted and real outcomes of the model. 

The **Evaluation** object contains many useful calls, such as *f1()*. This method esimates the accuracy of a model in the form of a probability (The f1 score below means the model considers its ability to classify about 77 percent accurate). Other methods include *precision()*, which tells us how well a model can reliably predict the same results given the same input, and *recall()* which tells us how many of the correct results we retrieved.

In this example, we have the following

     Actual Class 0 was predicted with Predicted 0 with count 50 times

     Actual Class 1 was predicted with Predicted 1 with count 1 times

     Actual Class 1 was predicted with Predicted 2 with count 49 times

     Actual Class 2 was predicted with Predicted 2 with count 50 times

    ====================F1Scores========================
                     0.767064393939394
    ====================================================

After your net has trained, you'll see an F1 score like this. In machine learning, an F1 score is one metric used to determine how well a classifier performs. It's a number between zero and one that explains how well the network performed during training. It is analogous to a percentage, with 1 being the equivalent of 100 percent predictive accuracy. It's basically the probability that your net's guesses are correct.

Our model wasn’t tuned all that well (back to the knobs!), and this is just a first pass, but not bad.

In the end, a working network's visual representation of the Iris dataset will look something like this

![Alt text](../img/iris_dataset.png)

