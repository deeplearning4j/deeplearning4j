---
title: Customized Data Pipelines for Loading Images Into Deep Neural Networks
layout: default
---

# Customized Data Pipelines for Images etc.

Deeplearning4j's examples run on benchmark datasets that don't present any obstacles in the data pipeline, because we abstracted them away. But real-world users start with raw, messy data, which they need to preprocess, vectorize and use to train a neural net for clustering or classification. 

*DataVec* is our machine-learning vectorization library, and it is useful for customizing how you prepare data that a neural net can learn. (The [DataVec Javadoc is here](http://deeplearning4j.org/datavecdoc/).)

This tutorial will walk through how to load an image dataset and carry out transforms on them. For the sake of simplicity this tutorial uses only 10 images from 3 of the classes in the *Oxford flower dataset*. Please do not copy paste the code below as they are only snippets for reference. 
[Use the code from the full example here](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataExamples/ImagePipelineExample.java)

## Setting up your images in the correct directory structure
In short, images in your dataset have to be organized in directories by class/label and these label/class directories live together in the parent directory.

* Download your dataset. 

* Make a parent directory.

* In your parent directory make subdirectories with names corresponding to the label/class names.

* Move all images belonging to a given class/label to it's corresponding directory.

Here is the directory structure expected in general.

>                                   parentDir
>                                 /   / | \  \
>                                /   /  |  \  \
>                               /   /   |   \  \
>                              /   /    |    \  \
>                             /   /     |     \  \
>                            /   /      |      \  \
>                      label_0 label_1....label_n-1 label_n


In this example the parentDir corresponds to `$PWD/src/main/resources/DataExamples/ImagePipeline/` and the subdirectories labelA, labelB, labelC all have 10 images each. 

## Specifying particulars before image load
* Specify path to the parent dir where the labeled images live in separate directories.
 
~~~java
File parentDir = new File(System.getProperty("user.dir"), "src/main/resources/DataExamples/ImagePipeline/");
~~~

* Specify allowed extensions and a random number generator to use when splitting dataset into test and train 

~~~java
FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
~~~

* Specifying a label maker so you do not have to manually specify labels. It will simply use the name of the subdirectories as label/class names.

~~~java
ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
~~~

* Specifying a path filter to gives you fine tune control of the min/max cases to load for each class. Below is a bare bones version. Refer to javadocs for details

~~~java
BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
~~~

* Specify your test train split, specified here as 80%-20%

~~~java
InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
InputSplit trainData = filesInDirSplit[0];
InputSplit testData = filesInDirSplit[1];
~~~

## Specifying particulars for your image pipeline transformation

* Specify an image record reader with height and width you want the entire dataset resized to. 

~~~java
ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
~~~
Please *note that the images in your dataset do not have to be the same size*. DataVec can do this for you. As you can see in this example images are all of different size and they are all resized to the height and width specified below

* Specify transformations

The advantage of neural nets is that they do not require you to manually feature engineer. However, there can be an advantage to transforming images to artificially increase the size of the dataset like in this winning kaggle entry <http://benanne.github.io/2014/04/05/galaxy-zoo.html>. Or you might want to crop an image down to only the relevant parts. An example would be detect a face and crop it down to size. DataVec has all the built in functionality/powerful features from OpenCV. A bare bones example that flips an image and then displays it is shown below:

~~~java
ImageTransform transform = new MultiImageTransform(randNumGen,new FlipImageTransform(), new ShowImageTransform("After transform"));
~~~

You can chain transformations as shown below, write your own classes that will automate whatever features you want.

~~~java
ImageTransform transform = new MultiImageTransform(randNumGen, new CropImageTransform(10), new FlipImageTransform(),new ScaleImageTransform(10), new WarpImageTransform(10));
~~~

* Initialize the record reader with the train data and the transform chain

~~~java
recordReader.initialize(trainData,transform);
~~~

## Handing off to fit
dl4j's neural net's take either a dataset or a dataset iterator to fit too. These are fundamental concepts for our framework. Please refer to other examples for how to use an iterator. Here is how you contruct a dataset iterator from an image record reader.

~~~java
DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum);
~~~

The DataSetIterator iterates through the input datasets via the recordReader, fetching one or more new examples with each iteration, and loading those examples into a DataSet object that neural nets can work with.
