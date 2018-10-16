---
title: "Deeplearning4j on Spark: How To Build Data Pipelines"
short_title: Spark Data Pipelines Guide
description: "Deeplearning4j on Spark: How To Build Data Pipelines"
category: Distributed Deep Learning
weight: 3
---

# Deeplearning4j on Spark: How To Build Data Pipelines

This page provides some guides on how to create data pipelines for both training and evaluation when using Deeplearning4j on Spark.

This page assumes some familiarity with Spark (RDDs, master vs. workers, etc) and Deeplearning4j (networks, DataSet etc).

As with training on a single machine, the final step of a data pipeline should be to produce a DataSet (single features arrays, single label array) or MultiDataSet (one or more feature arrays, one or more label arrays). In the case of DL4J on Spark, the final step of a data pipeline is data in one of the following formats:
(a) an ```RDD<DataSet>```/```JavaRDD<DataSet>```
(b) an ```RDD<MultiDataSet>```/```JavaRDD<MultiDataSet>```
(c) a directory of serialized DataSet/MultiDataSet (minibatch) objects on network storage such as HDFS, S3 or Azure blob storage
(d) a directory of minibatches in some other format

Once data is in one of those four formats, it can be used for training or evaluation.

**Note:** When training multiple models on a single dataset, it is best practice to preprocess your data once, and save it to network storage such as HDFS.
Then, when training the network you can call ```SparkDl4jMultiLayer.fit(String path)``` or ```SparkComputationGraph.fit(String path)``` where ```path``` is the directory where you saved the files.


Spark Data Prepration: How-To Guides
* [How to prepare a RDD[DataSet] from CSV data for classification or regression](#csv)
* [How to save a RDD[DataSet] or RDD[MultiDataSet] to network storage and use it for training](#saveloadrdd)
* [How to prepare data on a single machine for use on a cluster: saving DataSets](#singletocluster)
* [How to prepare data on a single machine for use on a cluster: map/sequence files](#singletocluster2)
* [How to load multiple CSVs (one sequence per file) for RNN data pipelines](#csvseq)
* [How to create an RDD[DataSet] for images](#images)
* [How to load prepared minibatches in custom format](#customformat)

<br><br>

## <a name="csv">How to prepare a RDD[DataSet] from CSV data for classification or regression</a>

This guide shows how to load data contained in one or more CSV files and produce a ```JavaRDD<DataSet>``` for export, training or evaluation on Spark.

The process is fairly straightforward. Note that the ```DataVecDataSetFunction``` is very similar to the ```RecordReaderDataSetIterator``` that is often used for single machine training.

For example, suppose the CSV had the following format - 6 total columns: 5 features followed by an integer class index for classification, and 10 possible classes

```
1.0,3.2,4.5,1.1,6.3,0
1.6,2.4,5.9,0.2,2.2,1
...
```

we could load this data for classification using the following code:
```
String filePath = "hdfs:///your/path/some_csv_file.csv";
JavaSparkContext sc = new JavaSparkContext();
JavaRDD<String> rddString = sc.textFile(filePath);
RecordReader recordReader = new CSVRecordReader(',');
JavaRDD<List<Writable>> rddWritables = rddString.map(new StringToWritablesFunction(recordReader));

int labelIndex = 5;         //Labels: a single integer representing the class index in column number 5
int numLabelClasses = 10;   //10 classes for the label
JavaRDD<DataSet> rddDataSetClassification = rddWritables.map(new DataVecDataSetFunction(labelIndex, numLabelClasses, false));
```

However, if this dataset was for regression instead, with again 6 total columns, 3 feature columns (positions 0, 1 and 2 in the file rows) and 3 label columns (positions 3, 4 and 5) we could load it using the same process as above, but changing the last 3 lines to:

```
int firstLabelColumn = 3;   //First column index for label
int lastLabelColumn = 5;    //Last column index for label
JavaRDD<DataSet> rddDataSetRegression = rddWritables.map(new DataVecDataSetFunction(firstColumnLabel, lastColumnLabel, true, null, null));
```

<br><br>

## <a name="saveloadrdd">How to save a RDD[DataSet] or RDD[MultiDataSet] to network storage and use it for training</a>

As noted at the start of this page, it is considered a best practice to preprocess and export your data once (i.e., save to network storage such as HDFS and reuse), rather than fitting from an ```RDD<DataSet>``` or ```RDD<MultiDataSet>``` directly in each training job.

There are a number of reasons for this:
* Better performance (avoid redundant loading/calculation): When fitting multiple models from the same dataset, it is faster to preprocess this data once and save to disk rather than preprocessing it again for every single training run.
* Minimizing memory and other resources: By exporting and fitting from disk, we only need to keep the DataSets we are currently using (plus a small async prefetch buffer) in memory, rather than also keeping many unused DataSet objects in memory. Exporting results in lower total memory use and hence we can use larger networks, larger minibatch sizes, or allocate fewer resources to our job.
* Avoiding recomputation: When an RDD is too large to fit into memory, some parts of it may need to be recomputed before it can be used (depending on the cache settings). When this occurs, Spark will recompute parts of the data pipeline multiple times, costing us both time and memory. A pre-export step avoids this recomputation entirely.

**Step 1: Saving**

Saving the DataSet objects once you have an ```RDD<DataSet>``` is quite straightforward:
```
JavaRDD<DataSet> rddDataSet = ...
int minibatchSize = 32;     //Minibatch size of the saved DataSet objects
String exportPath = "hdfs:///path/to/export/data";
JavaRDD<String> paths = rddDataSet.mapPartitionsWithIndex(new BatchAndExportDataSetsFunction(minibatchSize, exportPath), true);
```
Keep in mind that this is a map function, so no data will be saved until the paths RDD is executed - i.e., you should follow this with an operation such as:
```
paths.saveAsTextFile("hdfs:///path/to/text/file.txt");  //Specified file will contain paths/URIs of all saved DataSet objects
```
or
```
List<String> paths = paths.collect();    //Collection of paths/URIs of all saved DataSet objects
```
or
```
paths.foreach(new VoidFunction<String>() {
    @Override
    public void call(String path) {
        //Some operation on each path
    }
});
```


Saving an ```RDD<MultiDataSet>``` can be done in the same way using ```BatchAndExportMultiDataSetsFunction``` instead, which takes the same arguments.

**Step 2: Loading and Fitting**

The exported data can be used in a few ways.
First, it can be used to fit a network directly:
```
String exportPath = "hdfs:///path/to/export/data";
SparkDl4jMultiLayer net = ...
net.fit(exportPath);      //Loads the serialized DataSet objects found in the 'exportPath' directory
```
Similarly, we can use ```SparkComputationGraph.fitMultiDataSet(String path)``` if we saved an ```RDD<MultiDataSet>``` instead.


Alternatively, we can load up the paths in a few different ways, depending on if or how we saved them:

```
JavaSparkContext sc = new JavaSparkContext();

//If we used saveAsTextFile:
String saveTo = "hdfs:///path/to/text/file.txt";
paths.saveAsTextFile(saveTo);                         //Save
JavaRDD<String> loadedPaths = sc.textFile(saveTo);    //Load

//If we used collecting:
List<String> paths = paths.collect();                 //Collect
JavaRDD<String> loadedPaths = sc.parallelize(paths);  //Parallelize

//If we want to list the directory contents:
String exportPath = "hdfs:///path/to/export/data";
JavaRDD<String> loadedPaths = SparkUtils.listPaths(sc, exportPath);   //List paths using org.deeplearning4j.spark.util.SparkUtils
```

Then we can execute training on these paths by using methods such as ```SparkDl4jMultiLayer.fitPaths(JavaRDD<String>)```


<br><br>

## <a name="singletocluster">How to prepare data on a single machine for use on a cluster: saving DataSets</a>

Another possible workflow is to start with the data pipeline on a single machine, and export the DataSet or MultiDataSet objects for use on the cluster.
This workflow clearly isn't as scalable as preparing data on a cluster (you are using just one machine to prepare data) but it can be an easy option in some cases, especially when you have an existing data pipeline.

This section assumes you have an existing ```DataSetIterator``` or ```MultiDataSetIterator``` used for single-machine training. There are many different ways to create one, which is outside of the scope of this guide.

**Step 1: Save the DataSets or MultiDataSets**

Saving the contents of a DataSet to a local directory can be done using the following code:
```
DataSetIterator iter = ...
File rootDir = new File("/saving/directory/");
int count = 0;
while(iter.hasNext()){
  DataSet ds = iter.next();
  File outFile = new File(rootDir, "dataset_" + (count++) + ".bin");
  ds.save(outFile);
}
```
Note that for the purposes of Spark, the exact file names don't matter.
The process for saving MultiDataSets is almost identical.

As an aside: you can read these saved DataSet objects on a single machine (for non-Spark training) using [FileDataSetIterator](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/file/FileDataSetIterator.java)).

An alternative approach is to save directly to the cluster using output streams, to (for example) HDFS. This can only be done if the machine running the code is properly configured with the required libraries and access rights. For example, to save the DataSets directly to HDFS you could use:

```
JavaSparkContext sc = new JavaSparkContext();
FileSystem fileSystem = FileSystem.get(sc.hadoopConfiguration());
String outputDir = "hdfs:///my/output/location/";

DataSetIterator iter = ...
int count = 0;
while(iter.hasNext()){
  DataSet ds = iter.next();
  String filePath = outputDir + "dataset_" + (count++) + ".bin";
  try (OutputStream os = new BufferedOutputStream(fileSystem.create(new Path(outputPath)))) {
    ds.save(os);
  }
}
```


**Step 2: Load and Train on a Cluster**
The saved DataSet objects can then be copied to the cluster or network file storage (for example, using Hadoop FS utilities on a Hadoop cluster), and used as follows:
```
String dir = "hdfs:///data/copied/here";
SparkDl4jMultiLayer net = ...
net.fit(dir);      //Loads the serialized DataSet objects found in the 'dir' directory
```
or alternatively/equivalently, we can list the paths as an RDD using:
```
String dir = "hdfs:///data/copied/here";
JavaRDD<String> paths = SparkUtils.listPaths(sc, dir);   //List paths using org.deeplearning4j.spark.util.SparkUtils
```

<br><br>

## <a name="singletocluster2">How to prepare data on a single machine for use on a cluster: map/sequence files</a>

An alternative approach is to use Hadoop MapFile and SequenceFiles, which are efficient binary storage formats.
This can be used to convert the output of any DataVec ```RecordReader``` or ```SequenceRecordReader``` (including a custom record reader) to a format usable for use on Spark.
MapFileRecordWriter and MapFileSequenceRecordWriter require the following dependencies:
```
<dependency>
    <groupId>org.datavec</groupId>
    <artifactId>datavec-hadoop</artifactId>
    <version>${datavec.version}</version>
</dependency>
<dependency>
    <groupId>org.apache.hadoop</groupId>
    <artifactId>hadoop-common</artifactId>
    <version>${hadoop.version}</version>
    <!-- Optional exclusion for log4j in case you are using other logging frameworks -->
    <!--
    <exclusions>
        <exclusion>
            <groupId>log4j</groupId>
            <artifactId>log4j</artifactId>
        </exclusion>
        <exclusion>
            <groupId>org.slf4j</groupId>
            <artifactId>slf4j-log4j12</artifactId>
        </exclusion>
    </exclusions>
    -->
</dependency>
```

**Step 1: Create a MapFile Locally**
In the following example, a CSVRecordReader will be used, but any other RecordReader could be used in its place:
```
File csvFile = new File("/path/to/file.csv")
RecordReader recordReader = new CSVRecordReader();
recordReader.initialize(new FileSplit(csvFile));

//Create map file writer
String outPath = "/map/file/root/dir"
MapFileRecordWriter writer = new MapFileRecordWriter(new File(outPath));

//Convert to MapFile binary format:
RecordReaderConverter.convert(recordReader, writer);
```

The process for using a ```SequenceRecordReader``` combined with a ```MapFileSequenceRecordWriter``` is virtually the same.

Note also that ```MapFileRecordWriter``` and ```MapFileSequenceRecordWriter``` both support splitting - i.e., creating multiple smaller map files instead of creating one single (potentially multi-GB) map file. Using splitting is recommended when saving data in this manner for use with Spark.

**Step 2: Copy to HDFS or other network file storage**

The exact process is beyond the scope of this guide. However, it should be sufficient to simply copy the directory ("/map/file/root/dir" in the example above) to a location on HDFS.

**Step 3: Read and Convert to ```RDD<DataSet>``` for Training**

We can load the data for training using the following:
```
JavaSparkContext sc = new JavaSparkContext();
String pathOnHDFS = "hdfs:///map/file/directory";
JavaRDD<List<Writable>> rdd = SparkStorageUtils.restoreMapFile(pathOnHDFS, sc);     //import: org.datavec.spark.storage.SparkStorageUtils

//Note at this point: it's the same as the latter part of the CSV how-to guide
int labelIndex = 5;         //Labels: a single integer representing the class index in column number 5
int numLabelClasses = 10;   //10 classes for the label
JavaRDD<DataSet> rddDataSetClassification = rdd.map(new DataVecDataSetFunction(labelIndex, numLabelClasses, false));
```

<br><br>

## <a name="csvseq">How to load multiple CSVs (one sequence per file) for RNN data pipelines</a>

This guide shows how load CSV files for training an RNN.
The assumption is that the dataset is comprised of multiple CSV files, where:

* each CSV file represents one sequence
* each row/line of the CSV contains the values for one time step (one or more columns/values, same number of values in all rows for all files) 
* each CSV may contain a different number of lines to other CSVs (i.e., variable length sequences are OK here)
* header lines either aren't present in any files, or are present in all files

A data pipeline can be created using the following process:
```
String directoryWithCsvFiles = "hdfs:///path/to/directory";
JavaPairRDD<String, PortableDataStream> origData = sc.binaryFiles(directoryWithCsvFiles);

int numHeaderLinesEachFile = 0; //No header lines
int delimiter = ",";            //Comma delimited files
SequenceRecordReader seqRR = new CSVSequenceRecordReader(numHeaderLinesEachFile, delimiter);

JavaRDD<List<List<Writable>>> sequencesRdd = origData.map(new SequenceRecordReaderFunction(seqRR));

//Similar to the non-sequence CSV guide using DataVecDataSetFunction. Assuming classification here:
int labelIndex = 5;             //Index of the label column. Occurs at position/column 5
int numClasses = 10;            //Number of classes for classification
JavaRDD<DataSet> dataSetRdd = sequencesRdd.map(new DataVecSequenceDataSetFunction(labelIndex, numClasses, false));
```

<br><br>

## <a name="images">How to create an RDD[DataSet] for images</a>

This guide shows how to create an ```RDD<DataSet>``` for image classification, starting from images stored on a network file system such as HDFS.

Note that one limitation of the implementation is that the set of classes (i.e., the class/category labels when doing classification) needs to be known, provided or collected manually. This differs from using ImageRecordReader for classification on a single machine, which can automatically infer the set of class labels.

First, assume the images are in subdirectories based on their class labels. For example, suppose there are two classes, "cat" and "dog", the directory structure would look like:
```
rootDir/cat/img0.jpg
rootDir/cat/img1.jpg
...
rootDir/dog/img0.jpg
rootDir/dog/img1.jpg
...
```
(Note the file names don't matter in this example - however, the parent directory names are the class labels)

The data pipeline for image classification can be constructed as follows:
```
String rootDir = "hdfs://path/to/rootDir";
JavaPairRDD<String, PortableDataStream> files = sc.binaryFiles(rootDir);

List<String> labelsList = Arrays.asList("cat", "dog");  //Substitute your class labels here

int imageHW = 224;              //Height/width to convert images to
int imageChannels = 3;          //Image channels to convert images to - 3 = RGB, 1 = grayscale
ImageRecordReader imageRecordReader = new ImageRecordReader(imageHW, imageHW, imageChannels);
rr.setLabels(labelsList);

JavaRDD<List<Writable>> writablesRdd = origData.map(new RecordReaderFunction(rr));

int labelIndex = 1;   //Always index 1 with ImageRecordReader
int numClasses = labelsList.size();
JavaRDD<DataSet> dataSetRdd = writablesRdd.map(new DataVecDataSetFunction(labelIndex, numClasses, false));
```

And that's it.

Note 1: If a DataSetPreprocess is required (such as ```ImagePreProcessingScaler```) this can be provided as an argument to DataVecDataSetFunction:
```
JavaRDD<DataSet> dataSetRdd = writablesRdd.map(new DataVecDataSetFunction(labelIndex, numClasses, false, new ImagePreProcessingScaler(), null));
```

Note 2: for other label generation cases (such as labels provided from the filename instead of parent directory), or for tasks such as semantic segmentation, you can substitute a different PathLabelGenerator instead of the default. For example, if the label should come from the file name, you can use ```PatternPathLabelGenerator``` instead.
Let's say images are in the format "cat_img1234.jpg", "dog_2309.png" etc. We can use tho following process:
```
PathLabelGenerator labelGenerator = new PatternPathLabelGenerator("_", 0);  //Split on the "_" character, and take the first value
ImageRecordReader imageRecordReader = new ImageRecordReader(imageHW, imageHW, imageChannels, labelGenerator);
```

Note that PathLabelGenerator returns a Writable object, so for tasks like image segmentation, you can return an INDArray using the NDArrayWritable class in a custom PathLabelGenerator.

<br><br>

## <a name="customformat">How to load prepared minibatches in custom format</a>

DL4J Spark training supports the ability to load data serialized in a custom format. The assumption is that each file on the remote/network storage represents a single minibatch of data in some readable format.

Note that this approach is typically not required or recommended for most users, but is provided as an additional option for advanced users or those with pre-prepared data in a custom format or a format that is not natively supported by DL4J.
When files represent a single record/example (instead of a minibatch) in a custom format, a custom RecordReader could be used instead.

The interfaces of note are:

* [DataSetLoader](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-core/src/main/java/org/deeplearning4j/api/loader/DataSetLoader.java)
* [MultiDataSetLoader](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-core/src/main/java/org/deeplearning4j/api/loader/MultiDataSetLoader.java)

Both of which extend the single-method [Loader](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-common/src/main/java/org/nd4j/api/loader/Loader.java) interface.

Suppose a HDFS directory contains a number of files, each being a minibatch in some custom format.
These can be loaded using the following process:
```
JavaSparkContext sc = new JavaSparkContext();
String dataDirectory = "hdfs:///path/with/data";
JavaRDD<String> loadedPaths = SparkUtils.listPaths(sc, dataDirectory);   //List paths using org.deeplearning4j.spark.util.SparkUtils

SparkDl4jMultiLayer net = ...
Loader<DataSet> myCustomLoader = new MyCustomLoader();
net.fitPaths(loadedPaths, myCustomLoader);
```

Where the custom loader class looks something like:
```
public class MyCustomLoader implements DataSetLoader {
    @Override
    public DataSet load(Source source) throws IOException {
        InputStream inputStream = source.getInputStream();
        <load custom data format here> 
        INDArray features = ...;
        INDArray labels = ...;
        return new DataSet(features, labels);
    }
}
```
