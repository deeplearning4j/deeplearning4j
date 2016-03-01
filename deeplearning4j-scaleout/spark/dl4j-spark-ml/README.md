
# Deeplearning4j on Spark ML

*Status: Experimental (all interfaces are subject to change)*

Spark 1.4 provides a standardized API for machine learning called [Spark ML](http://spark.apache.org/docs/latest/ml-guide.html).   DL4J enhances Spark ML with powerful neural network algorithms for supervised and unsupervised learning. Benefits include easy integration of DL4J with Spark-based datasets, with other Spark ML components (such as feature extractors and learning algorithms), and with Spark SQL.

## Getting Started

### Install Spark 1.4.x
Download and install Spark 1.4.x from [spark.apache.org](http://spark.apache.org/downloads.html).

#### Mac OS X
On a Mac, is it recommended that the [Homebrew package manager](http://brew.sh/) be used to install Spark.  After installing Homebrew, install Spark:
```
brew install apache-spark
```

#### Linux
Download and unpack Spark to a location such as `/opt/spark-1.4.x`.  Add the location to your `PATH`.

### Install dl4j-spark-ml Library
The `dl4j-spark-ml` library is not yet published to the Maven Central repository and must be built from source.
```
git clone https://github.com/deeplearning4j/deeplearning4j.git
cd deeplearning4j
mvn clean install -DskipTests
```
### Install dl4j-spark-ml Spark Package
The 'dl4j-spark-ml' Spark Package is not yet published to the Spark Packages repository and must be built from source.
```
git clone https://github.com/deeplearning4j/dl4j-spark-ml.git
cd dl4j-spark-ml
sbt publishM2
```

## Using DL4J with Spark ML

We currently encourage everyone to use versions of Spark that are not Spark ML. [Examples](https://github.com/deeplearning4j/dl4j-spark-cdh5-examples).

### Data Sources

Spark ML supports operating on a variety of data sources through the DataFrame interface.  Load your data into a DataFrame by any means, including:

1. Use a built-in data source, such as Parquet, JSON, JDBC, or Hive.  See [the documentation](http://spark.apache.org/docs/latest/sql-programming-guide.html#data-sources).
2. Develop a Spark SQL relation provider.  See the [announcement](https://databricks.com/blog/2015/01/09/spark-sql-data-sources-api-unified-data-access-for-the-spark-platform.html), the [Iris relation](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/spark/dl4j-spark-ml/src/main/scala/org/deeplearning4j/spark/sql/sources/iris), and the [LFW relation](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/spark/dl4j-spark-ml/src/main/scala/org/deeplearning4j/spark/sql/sources/lfw).  More information will be available soon.

Data may be loaded from any Hadoop-compatible filesystem.  *We recommend that you load datasets from HDFS rather than from the local filesystem.*

#### Data Types
The DL4J pipeline components expect feature data to be provided as an ML column of [Vectors](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.linalg.Vectors$).   The data should be normalized; consider using [StandardScaler](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.StandardScaler) in your pipeline.

In supervised learning scenarios, labels are expected to be provided as an ML column of label indices.   Consider using [StringIndexer](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.StringIndexer) in your pipeline.

### ML Algorithms
Spark ML standardizes APIs for machine learning algorithms (see the appendix).  The `dl4j-spark-ml` library provides a number of transformers to include in an ML pipeline.

Most deeplearning4j neural network-based algorithms may be used with Spark ML.   Simply provide an instance of `org.deeplearning4j.nn.conf.MultiLayerConfiguration` as a parameter to the transformer.   Visit [deeplearning4j.org](http://deeplearning4j.org) for more information on the core functionality.

#### Classification
For supervised classification scenarios, use `o.d.spark.ml.classification.NeuralNetworkClassification` as a pipeline component.  Extends [Classifier](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.Classifier) and supports multi-class labels.

The following pipeline parameters are provided:

| Parameter        | Description |
| ---------------- | ----------- |
| conf             | The MultiLayerConfiguration instance that describes the neural network. |
| epochs           | The number of full passes over the dataset, with convergence on each pass.  Default is 1. |
| featuresCol      | The name of the column containing feature data. Default is 'features'. |
| labelCol         | The name of the column containing label data.  Default is 'label'. |
| predictionCol    | The name of the column containing predicted labels. Default is 'prediction'. | 
| rawPredictionCol | The name of the column containing raw probabilities for each label. | 

When configuring the neural network, ensure that the # of input columns on the input layer matches the expected size of the feature vector.  The # of output columns on the output layer should match the expected number of labels/classes.  Tip: if using `StringIndexer`, set the number of output columns to zero to have the classifier adjust the configuation automatically.

#### Reconstruction
For reconstruction scenarios such as deep auto-encoding, use `o.d.spark.ml.reconstruction.NeuralNetworkReconstruction` as a pipeline component.  The component performs unsupervised learning for which no label is necessary.

The following pipeline parameters are provided:

| Parameter         | Description |
| ----------------- | ----------- |
| conf              | The MultiLayerConfiguration instance that describes the neural network. |
| epochs            | The number of full passes over the dataset, with convergence on each pass.  Default is 1. |
| layerIndex        | The layer from which to draw reconstruction data.  Default is 1 (the input layer). |
| reconstructionCol | The name of the column containing reconstruction data.  Default is 'reconstruction'. |

When configuring the neural network, ensure that the # of input columns on the input layer matches the expected size of the feature vector.  

### Evaluation
Spark ML provides a standardized evaluation API for scoring and model selection.   See the [Programming Guide](http://spark.apache.org/docs/latest/ml-guide.html#example-model-selection-via-cross-validation) for more information.

## Appendix: Spark Concepts
Spark ML introduces a new API for machine learning based on powerful Spark SQL abstractions.  Please read the [Spark ML Programming Guide](http://spark.apache.org/docs/latest/ml-guide.html) for more information.

### DataFrame
A DataFrame is a distributed collection of data organized into named columns, with an associated schema.   Under the hood, a data frame has an associated `RDD[Row]`. Through the familiar RDD transformation process, columns may be added to and deleted from derived data frames.  

#### Columns
Each column of a DataFrame has an associated name, data type, and optional metadata.  

Spark ML defines the notion of 'ML column'.  An ML column is of a certain subset of data types - Vector, numeric data, nominal data - and contains metadata such as label strings, # of unique labels, etc.

#### Data Types
For ML purposes, the following data types are preferred for ML columns:

| Data Type | Description |
| --------- | ----------- |
| Vector    | A sparse or dense array of doubles |
| Double    | A numeric label |

Strings should be converted to an ML type using a transformer such as `StringIndexer`.  Please consider LabeledPoint to be obsolete.

#### Data Sources
Spark SQL provides a sophisticated set of abstractions for loading data from any source.   Capabilties include column pruning (to load only those columns needed for a given transformation) and predicate push-down.

DL4J provides the following data sources for reading certain well-known datasets:

| Source | Description |
| --------- | ------- |
| LFW | Labeled Faces in the Wild (LFW) dataset as a DataFrame. |
| Iris | Iris dataset as a DataFrame. |

### ML Pipeline

ML pipelines are based on the idea of learning from data in DataFrame columns and appending prediction columns.  Please read the [documentation](http://spark.apache.org/docs/latest/ml-guide.html).

#### Estimators
An estimator is a learning algorithm, expected to produce a model based on provided input.  The estimator does not typically alter the input DataFrame.

#### Transformers
A transformer alters a data frame, typically appending a column based on a UDF over existing column(s).

#### Models
A model is a type of transformer that appends predictions, reconstructions, anomaly indicators, etc.

#### Evaluators
An evaluator facilitates cross-fit validation based on an evaluation algorithm.

#### Pipeline
The ML pipeline consists of a DAG of estimators and transformers.  Data frames pass through the pipeline, being augmented with machine learning predictions, reconstructions etc., while retaining the application-specific context in other columns. 
