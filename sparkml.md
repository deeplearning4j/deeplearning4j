---
title: 
layout: default
---

# Deeplearning4j for Spark ML

We’re integrating with the Spark ML pipeline leveraging the developer API. The integration code is located in the [dl4j-spark-ml module](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/spark/dl4j-spark-ml) in the deeplearning4j repository.

Major aspects of the integration work:

1. **ML algorithms.** To bind the dl4j algorithms to the ML pipeline, we developed a [new classifier](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-scaleout/spark/dl4j-spark-ml/src/main/scala/org/deeplearning4j/spark/ml/classification/MultiLayerNetworkClassification.scala) and a new [unsupervised learning estimator](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-scaleout/spark/dl4j-spark-ml/src/main/scala/org/deeplearning4j/spark/ml/Unsupervised.scala).   
2. **ML attributes.** We strove to interoperate well with other pipeline components. ML Attributes are column-level metadata enabling information sharing between pipeline components. See [here how the classifier reads label metadata](https://github.com/deeplearning4j/deeplearning4j/blob/4d33302dd8a792906050eda82a7d50ff77a8d957/deeplearning4j-scaleout/spark/dl4j-spark-ml/src/main/scala/org/deeplearning4j/spark/ml/classification/MultiLayerNetworkClassification.scala#L89) from a column provided by the new [StringIndexer](http://people.apache.org/~pwendell/spark-releases/spark-1.4.0-rc4-docs/api/scala/index.html#org.apache.spark.ml.feature.StringIndexer).
3. **Large binary data.** It is challenging to work with large binary data in Spark. An effective approach is to leverage PrunedScan and to carefully control partition sizes. [Here](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-scaleout/spark/dl4j-spark-ml/src/main/scala/org/deeplearning4j/spark/sql/sources/lfw/LfwRelation.scala) we explored this with a custom data source based on the new relation API.   
4. **Column-based record readers.** [Here](https://github.com/deeplearning4j/deeplearning4j/blob/b237385b56d42d24bd3c99d1eece6cb658f387f2/deeplearning4j-scaleout/spark/dl4j-spark-ml/src/main/scala/org/deeplearning4j/spark/sql/sources/lfw/LfwRelation.scala#L96) we explored how to construct rows from a Hadoop input split by composing a number of column-level readers, with pruning support.
5. **UDTs.** With Spark SQL it is possible to introduce new data types. We prototyped an experimental Tensor type, [here](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-scaleout/spark/dl4j-spark-ml/src/main/scala/org/deeplearning4j/spark/sql/types/tensors.scala).
6. **Spark Package.** We developed a spark package to make it easy to use the dl4j framework in spark-shell and with spark-submit. See the [deeplearning4j/dl4j-spark-ml](https://github.com/deeplearning4j/dl4j-spark-ml) repository for useful snippets involving the sbt-spark-package plugin.
7. **Example code.** Examples demonstrate how the standardized ML API simplifies interoperability, such as with label preprocessing and feature scaling. See the [deeplearning4j/dl4j-spark-ml-examples](https://github.com/deeplearning4j/dl4j-spark-ml-examples) repository for an expanding set of example pipelines.
Hope this proves useful to the community as we transition to exciting new concepts in Spark SQL and Spark ML. Meanwhile, we have Spark working with [multiple GPUs on AWS](http://deeplearning4j.org/gpu_aws.html) and we're looking forward to optimizations that will speed neural net training even more. 

*~Eron Wright*
