# dl4j-streaming

This project combines a few pieces of technology for streaming applicaitons such as video or anomaly detection/web logs.

1. [Camel](http://camel.apache.org)
2. [Kafka](http://kafka.apache.org)
3. [Spark Streaming with kafka](http://spark.apache.org/docs/latest/streaming-kafka-integration.html)
4. [DataVec](https://github.com/deeplearning4j/DataVec)

The goal of these projects is to leverage the [components from camel](http://camel.apache.org/components.html)
with the machine learning specific dsl of DataVec to be able to handle ETL with different systems
in a streaming environment. The key here is the [DataVec and Camel integration](https://github.com/deeplearning4j/DataVec/tree/master/datavec-camel)
which handles dealing with [camel routes](http://camel.apache.org/architecture.html) and using a [DataVec marshaller](https://github.com/deeplearning4j/DataVec/blob/master/datavec-camel/src/main/java/org/datavec/camel/component/csv/marshaller/ListStringInputMarshaller.java)
you can plug in to a camel route and convert things to records. 

This happens via  a [Camel processor](http://camel.apache.org/processor.html) 

From there you would then turn the records in to NDArrays for dl4j-spark to consume.
