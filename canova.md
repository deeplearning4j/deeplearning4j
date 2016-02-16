---
title: "Canova: A Vectorization Lib for Machine Learning"
layout: default
---

# Canova: A Vectorization Lib for ML

Canova solves one of the most important obstacles to effective machine or deep learning: getting data into a format that neural nets can understand. Nets understand vectors. Vectorization is the first problem many data scientists will have to solve to start training their algorithms on data. Please read this entire page, particularly the section [Reading Records](#record) below, before working with Canova.

## Introductory Video

<iframe width="420" height="315" src="https://www.youtube.com/embed/EHHtyRKQIJ0" frameborder="0" allowfullscreen></iframe>

## Key Aspects
- [Canova](https://github.com/deeplearning4j/Canova) uses an input/output format system (similar to how Hadoop uses MapReduce)
- Designed to support all major types of input data (text, CSV, audio, image and video) with these specific input formats
- Uses an output format system to specify an implementation-neutral type of vector format (ARFF, SVMLight, etc.)
- Runs from the command line
- Can be extended for specialized input formats (such as exotic image formats); i.e. You can write your own custom input format and let the rest of the codebase handle the transformation pipeline
- Makes vectorization a first-class citizen
- Please see the [Canova Javadoc](http://deeplearning4j.org/canovadoc/) here

We're finishing up the command-line interface (CLI) system and running tests on basic datasets for converting stock CSV data that you'd export from a database or download from UCI. You can transform this data with a small conf file-based transform language (label, normalize, copy, etc). There's a <a href="#tutorial">brief tutorial below</a>.

## A Few Examples

 * Convert the CSV-based UCI Iris dataset into svmLight open vector text format
 * Convert the MNIST dataset from raw binary files to the svmLight text format.
 * Convert raw text into the Metronome vector format
 * Convert raw text into TF-IDF based vectors in a text vector format {svmLight, metronome, arff}
 * Convert raw text into the word2vec in a text vector format {svmLight, metronome, arff}

## Targeted Vectorization Engines

 * Any CSV to vectors with a scriptable transform language
 * MNIST to vectors
 * Text to vectors
    * TF-IDF
    * Bag of Words
    * word2vec

## CSV Transformation Engine

Below is an example of the CSV transform language in action from the command line

### UCI Iris Schema Transform

    @RELATION UCIIrisDataset
    @DELIMITER ,
       @ATTRIBUTE sepallength  NUMERIC   !NORMALIZE
       @ATTRIBUTE sepalwidth   NUMERIC   !NORMALIZE
       @ATTRIBUTE petallength  NUMERIC   !NORMALIZE
       @ATTRIBUTE petalwidth   NUMERIC   !NORMALIZE
       @ATTRIBUTE class        STRING   !LABEL

## <a name="tutorial">Setting Up Canova</a>

Search for [canova](https://search.maven.org/#search%7Cga%7C1%7CCanova) on Maven Central to get a list of JARs you can use.

Add the dependency information into your pom.xml.

<!-- You'll need to do a *git clone* from [Canova's Github repo](https://github.com/deeplearning4j/Canova), and then build the dependencies with [Maven](http://nd4j.org/getstarted.html#maven). 

      mvn -DskipTests=true -Dmaven.javadoc.skip=true install

(We also recommend that you clone the [ND4J repo](https://github.com/deeplearning4j/nd4j) and build its dependencies now.)

Then you'll want to build the stand-alone Canova jar to run the CLI from terminal/command prompt:

      cd canova-cli/
      mvn -DskipTests=true -Dmaven.javadoc.skip=true package


## Create the Configuration File

You'll need a file to tell the vectorization engine what to do. Create a text file containing the following lines in the *canova-cli* directory (you might name the file *vec_conf.txt*):

    input.header.skip=false
    input.statistics.debug.print=false
    input.format=org.canova.api.formats.input.impl.LineInputFormat
    
    input.directory=src/test/resources/csv/data/uci_iris_sample.txt
    input.vector.schema=src/test/resources/csv/schemas/uci/iris.txt
    output.directory=/tmp/iris_unit_test_sample.txt
    
    output.format=org.canova.api.formats.output.impl.SVMLightOutputFormat

-->
<!--## Run Canova From the Command Line

Now we're going to take this [sample](https://github.com/deeplearning4j/Canova/blob/master/canova-cli/src/test/resources/csv/data/uci_iris_sample.txt) of [UCI's Iris dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).

    5.1,3.5,1.4,0.2,Iris-setosa
    4.9,3.0,1.4,0.2,Iris-setosa
    4.7,3.2,1.3,0.2,Iris-setosa
    7.0,3.2,4.7,1.4,Iris-versicolor
    6.4,3.2,4.5,1.5,Iris-versicolor
    6.9,3.1,4.9,1.5,Iris-versicolor
    5.5,2.3,4.0,1.3,Iris-versicolor
    6.5,2.8,4.6,1.5,Iris-versicolor
    6.3,3.3,6.0,2.5,Iris-virginica
    5.8,2.7,5.1,1.9,Iris-virginica
    7.1,3.0,5.9,2.1,Iris-virginica
    6.3,2.9,5.6,1.8,Iris-virginica

Transform it into the svmLight format from the command line like this

    ./bin/canova vectorize -conf [my_conf_file]

The output in your command prompt should look like

    ./bin/canova vectorize -conf /tmp/iris_conf.txt 
    File path already exists, deleting the old file before proceeding...
    Output vectors written to: /tmp/iris_svmlight.txt

If you *cd* into */tmp* and open *iris_svmlight.txt*, you'll see something like this:

    0.0 1:0.1666666666666665 2:1.0 3:0.021276595744680823 4:0.0
    0.0 1:0.08333333333333343 2:0.5833333333333334 3:0.021276595744680823 4:0.0
    0.0 1:0.0 2:0.7500000000000002 3:0.0 4:0.0
    1.0 1:0.9583333333333335 2:0.7500000000000002 3:0.723404255319149 4:0.5217391304347826
    1.0 1:0.7083333333333336 2:0.7500000000000002 3:0.6808510638297872 4:0.5652173913043479
    1.0 1:0.916666666666667 2:0.6666666666666667 3:0.7659574468085107 4:0.5652173913043479
    1.0 1:0.3333333333333333 2:0.0 3:0.574468085106383 4:0.47826086956521746
    1.0 1:0.7500000000000001 2:0.41666666666666663 3:0.702127659574468 4:0.5652173913043479
    2.0 1:0.6666666666666666 2:0.8333333333333333 3:1.0 4:1.0
    2.0 1:0.45833333333333326 2:0.3333333333333336 3:0.8085106382978723 4:0.7391304347826088
    2.0 1:1.0 2:0.5833333333333334 3:0.9787234042553192 4:0.8260869565217392
    2.0 1:0.6666666666666666 2:0.5 3:0.9148936170212765 4:0.6956521739130436

## Feeding Vectors Into Deeplearning4j

Deeplearning4j also works with a command-line interface. A net can be trained with the following script, drawing on the vectorized input you just created with Canova:

      ./bin/deeplearning4j train -input input/file/path/tmp/iris_svmlight.txt -output output/file/path/output.txt -runtime hadoop -model modelConfig.java
      
The configuration of the net itself may need to be adjusted within the file that contains its instantiation and parameters. Examples of these configurations can be seen on the pages describing [restricted Boltzmann machines](http://deeplearning4j.org/restrictedboltzmannmachine.html) as well as the [Mnist tutorial](http://deeplearning4j.org/mnist-tutorial.html). 
-->

## <a name="record">Reading Records, Iterating Over Data</a>

The following code shows how to work with one example, raw images, transforming them into a format that will work well with DL4J and ND4J:

        // Instantiating RecordReader. Specify height and width of images.
        RecordReader recordReader = new ImageRecordReader(28, 28, true, labels);

        // Point to data path. 
        recordReader.initialize(new FileSplit(new File(labeledPath)));

The RecordReader is a class in Canova that helps convert the byte-oriented input into data that's oriented toward a record; i.e. a collection of elements that are fixed in number and indexed with a unique ID. Converting data to records is the process of vectorization. The record itself is a vector, each element of which is a feature.

The [ImageRecordReader](https://github.com/deeplearning4j/Canova/blob/f03f32dd42f14af762bf443a04c4cfdcc172ac83/canova-nd4j/canova-nd4j-image/src/main/java/org/canova/image/recordreader/ImageRecordReader.java) is a subclass of the RecordReader and is built to automatically take in 28 x 28 pixel images. Thus, LFW images are scaled to 28 pixels x 28 pixels. You can change dimensions to match your custom images by changing the parameters fed to the ImageRecordReader, as long as you make sure to adjust the `nIn` hyperparameter, which will be equal to the product of image height x image width. 

Other parameters shown above include `true`, which instructs the reader to append a label to the record, and `labels`, which is the array of supervised values (e.g. targets) used to validate neural net model results. Here are all the RecordReader extensions that come pre-built with Canova (you can find them by right-clicking on `RecordReader` in IntelliJ, clicking `Go To` in the drop-down menu, and selection `Implementations`):

![Alt text](../img/recordreader_extensions.png)

The DataSetIterator is a Deeplearning4J class that traverses the elements of a list. Iterators pass through the data list, accesses each item sequentially, keeps track of how far it has progressed by pointing to its current element, and modifies itself to point to the next element with each new step in the traversal.

        // Canova to DL4J
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 784, labels.size());

The DataSetIterator iterates through input datasets, fetching one or more new examples with each iteration, and loading those examples into a DataSet object that neural nets can work with. The line above also tells the [RecordReaderDataSetIterator](https://github.com/deeplearning4j/deeplearning4j/blob/3e5c6a942864ced574c7715ae548d5e3cb22982c/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/canova/RecordReaderDataSetIterator.java) to convert the image to a straight line (e.g. vector) of elements, rather than a 28 x 28 grid (e.g. matrix); it also specifies the number of labels possible.

`RecordReaderDataSetIterator` can take as parameters the specific recordReader you want (for images, sound, etc.) and the batch size. For supervised learning, it will also take a label index and the number of possible labels that can be applied to the input (for LFW, the number of labels is 5,749). 

For a walkthrough of the other steps associated with moving data from Canova to Deeplearning4j, you can read about [how to build a customized image data pipeline here](../image-data-pipeline.html).

## Execution

Runs as both a local serial process and a MapReduce (MR engine on the roadmap) scale-out process with no code changes.

## Targetted Vector Formats
* svmLight
* libsvm
* Metronome
* ARFF

## Built-In General Functionality
* Understands how to take general text and convert it into vectors with stock techniques such as kernel hashing and TF-IDF
