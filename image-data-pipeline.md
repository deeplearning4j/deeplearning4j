---
title: Customized Data Pipelines for Loading Images Into Deep Neural Networks
layout: default
---

# Customized Data Pipelines for Images etc.

Deeplearning4j's examples run on benchmark datasets that don't present any obstacles in the data pipeline, because we abstracted them away. But real-world users start with raw, messy data, which they need to preprocess, vectorize and use to train a neural net for clustering or classification. 

*Canova* is our machine-learning vectorization library, and it is useful for customizing how you prepare data that a neural net can learn. (The [Canova Javadoc is here](http://deeplearning4j.org/canovadoc/).)

This tutorial will walk through how it loads *Labeled Faces in the Wild*, a supervised set of 13,233 photographs representing 5,749 relatively famous people.

## Introductory Video

<iframe width="420" height="315" src="https://www.youtube.com/embed/EHHtyRKQIJ0" frameborder="0" allowfullscreen></iframe>

## Loading Labels

Download the LFW dataset and place it in the right file path (i.e. location in your computer's directories). Use the following code to feed the path into a variable called labelPath. Now you're ready to read and load the data, and create an array to hold the images' labels.

        // Set path to the labeled images
        String labeledPath = System.getProperty("user.home")+"/lfw";
        
        //create array of strings called labels
         List<String> labels = new ArrayList<>(); 
        
        //traverse dataset to get each label
        for(File f : new File(labeledPath).list Files()) { 
            labels.add(f.getName());
        }

## <a name="record">Reading Records, Iterating Over Data</a>

The following code helps transform raw images into a format that will work well with DL4J and ND4J:

        // Instantiating RecordReader. Specify height and width of images.
        RecordReader recordReader = new ImageRecordReader(28, 28, true, labels);

        // Point to data path. 
        recordReader.initialize(new FileSplit(new File(labeledPath)));

The RecordReader is a class in Canova that helps convert the byte-oriented input into data that's oriented toward a record; i.e. a collection of elements that are fixed in number and indexed with a unique ID. Converting data to records is the process of vectorization. The record itself is a vector, each element of which is a feature.

The [ImageRecordReader](https://github.com/deeplearning4j/Canova/blob/master/canova-nd4j/canova-nd4j-image/src/main/java/org/canova/image/recordreader/ImageRecordReader.java) is a subclass of the RecordReader and is built to automatically take in 28 x 28 pixel images. Thus, LFW images are scaled to 28 pixels x 28 pixels. You can change dimensions to match your custom images by changing the parameters fed to the ImageRecordReader, as long as you make sure to adjust the `nIn` hyperparameter, which will be equal to the product of image height x image width. 

Other parameters shown above include `true`, which instructs the reader to append a label to the record, and `labels`, which is the array of supervised values (e.g. targets) used to validate neural net model results. Here are all the RecordReader extensions that come pre-built with Canova (you can find them by right-clicking on `RecordReader` in IntelliJ, clicking `Go To` in the drop-down menu, and selection `Implementations`):

![Alt text](../img/recordreader_extensions.png)

The DataSetIterator is a Deeplearning4J class that traverses the elements of a list. Iterators pass through the data list, accesses each item sequentially, keeps track of how far it has progressed by pointing to its current element, and modifies itself to point to the next element with each new step in the traversal.

        // Canova to DL4J
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 784, labels.size());

The DataSetIterator iterates through input datasets, fetching one or more new examples with each iteration, and loading those examples into a DataSet object that neural nets can work with. The line above also tells the [RecordReaderDataSetIterator](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/canova/RecordReaderDataSetIterator.java) to convert the image to a straight line (e.g. vector) of elements, rather than a 28 x 28 grid (e.g. matrix); it also specifies the number of labels possible.

`RecordReaderDataSetIterator` can take as parameters the specific recordReader you want (for images, sound, etc.) and the batch size. For supervised learning, it will also take a label index and the number of possible labels that can be applied to the input (for LFW, the number of labels is 5,749). 

## Configuring the Model

Below is a neural net configuration example. Many of the hyperparameters have been explained in the [Iris tutorial](../iris-flower-dataset-tutorial.html); thus, we'll summarize a few distinguishing characteristics.

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/deepbelief/DBNIrisExample.java?slice=64:98"></script>

* *optimizationAlgo* relies on conjugate gradient, rather than LBFGS. 
* *nIn* is set to 784, so each image pixel becomes an input node. If your images change dimensions (meaning more or less pixels total), nIn should, too.
* *list* operator is set to 4, which means three RBM hidden layers and one output layer. More than one RBM becomes a DBN.
* *lossFunction* is set to Monte Carlo Cross Entropy. This loss function will be used to train the classification layer. 

## Building and Training the Model

At the end of the configuration, call build and pass the network's configuration into a MultiLayerNetwork object.

                }).build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);

To set iteration listeners that show performance and help tune while training the neural net, use one of these examples:

        network.setListeners(Arrays.<IterationListener>asList(new ScoreIterationListener(10), new GradientPlotterIterationListener(10)));

        OR

        network.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

## Training the Model

Once the data is loaded, the model framework is built, train the model to fit the data. Call next on the data iterator to advance it through the data based on the batch size. It will return a certain amount of data based on batch size each time. The code below shows how to loop through the dataset iterator and then to run fit on the model in order to train it on the data.

        // Training
        while(iter.hasNext()){
            DataSet next = iter.next();
            network.fit(next);
        }

## Evaluating the Model

After the model has been trained, run data through it to test and evaluate its performance. Typically its a good idea to use cross validation by splitting up the dataset and using data the model hasn't seen before. In this case we have just show you below how to reset the current iterator, initialize the evaluation object and run data through it to get performance information.

        // Using the same training data as test. 
        
        iter.reset();
        Evaluation eval = new Evaluation();
        while(iter.hasNext()){
            DataSet next = iter.next();
            INDArray predict2 = network.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), predict2);
        }
        
        System.out.println(eval.stats());

An alternative approach to apply cross validation in this effort, would be to load all the data and split it up into a train and test set. Iris is a small enough dataset to load all the data and accomplish the split. Many datasets used for production neural nets are not. For the alternative approach in this example, use the following code:

        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum, new Random(seed));
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();

To split test and train on large datasets, you'll have to iterate through both the test and training datasets. For the moment, we'll leave that to you. 
