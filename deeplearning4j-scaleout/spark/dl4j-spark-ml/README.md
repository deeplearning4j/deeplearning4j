
# DL4J on Spark ML

Spark 1.2+ provides a standardized API for machine learning called [Spark ML](http://spark.apache.org/docs/latest/ml-guide.html).   DL4J has experimental support for Spark ML.  The main benefit is easy integration of DL4J with Spark data sources, with other ML components (such as feature extractors and learning algorithms), and with SQL.

## Caveats
Be aware that this module will undergo breaking changes for Spark 1.4.  The DL4J-on-Spark-ML classes are located at `org.apache.spark.ml` to work around a limitation of Spark 1.3, that the Spark ML extensibilty API is currently marked 'private'.   For 1.4, we expect to move these classes to 'org.deeplearning4j.spark.ml'.

# Getting Started

## System Requirements
- Please use Spark 1.3.

## Examples
Please try the examples at [deeplearning4j/spark-examples](https://github.com/deeplearning4j/spark-examples).  ML-related examples:

- `ml.JavaIrisClassificationPipeline`

# Concepts
Spark ML introduces a new API for machine learning based on powerful Spark SQL abstractions (see [the docs](http://spark.apache.org/docs/latest/sql-programming-guide.html)).

## DataFrame
A DataFrame is a distributed collection of data organized into named columns, with an associated schema.   Under the hood, a data frame has an associated `RDD[Row]`. Through the familiar RDD transformation process, columns may be added to and deleted from derived data frames.  

### Columns
Each column has an associated name, data type, and optional metadata.  

In Spark 1.4, a formal notion of 'ML column' will be introduced.  An ML column is of a certain subset of datatypes - Vector, numeric data, nominal data - and contains metadata such as label strings, # of unique labels, etc.

### Data Types
For ML purposes, the following data types are preferred for ML columns:

| Data Type | Description |
| --------- | ----------- |
| Vector    | A sparse or dense array of doubles |
| Double    | A numeric label |
| ~~String~~    | Strings are not expected in ML columns! |

Please consider LabeledPoint to be obsolete.

## ML Pipeline

ML pipelines are based on the idea of learning from data in DataFrame columns and appending prediction columns.  Please read the [documentation](http://spark.apache.org/docs/latest/ml-guide.html).

### Estimators
An estimator is a learning algorithm, expected to produce a model based on provided input.  The estimator does not typically alter the input DataFrame.

DL4J provides:

| Estimator | Purpose |
| --------- | ------- |
| NeuralNetworkClassification | Neural network-based learning algorithm for supervised classification.  Supports multi-class labels. |
| NeuralNetworkReconstruction | Neural network-based learning algorithm for unsupervised reconstruction. |

### Transformers
A transformer alters a data frame, typically appending a column based on a UDF over existing column(s).

### Model
A model is a type of transformer that appends predictions, reconstructions, anomaly indicators, etc.

### Evaluators
An evaluator facilitates cross-fit validation based on an evaluation algorithm.

_DL4J has functionality here that is not yet exposed to Spark ML._

### Pipeline
The ML pipeline consists of a DAG of estimators and transformers.  Data frames pass through the pipeline, being augmented with machine learning predictions, reconstructions etc., while retaining the application-specific context in other columns. 



