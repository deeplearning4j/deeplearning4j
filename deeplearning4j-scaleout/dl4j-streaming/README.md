# dl4j-streaming

This project combines a few pieces of technology for streaming applicaitons such as video or anomaly detection/web logs.

1. [Camel](http://camel.apache.org)
2. [Kafka](http://kafka.apache.org)
3. [Spark Streaming with kafka](http://spark.apache.org/docs/latest/streaming-kafka-integration.html)
4. [Canova](https://github.com/deeplearning4j/Canova)

The goal of these projects is to leverage the [components from camel](http://camel.apache.org/components.html)
with the machine learning specific dsl of canova to be able to handle ETL with different systems
in a streaming environment. The key here is the [canova and camel integration](https://github.com/deeplearning4j/Canova/tree/master/canova-camel)
which handles dealing with [camel routes](http://camel.apache.org/architecture.html) and using a [canova marshaller](https://github.com/deeplearning4j/Canova/blob/master/canova-camel/src/main/java/org/canova/camel/component/csv/marshaller/ListStringInputMarshaller.java)
you can plug in to a camel route and convert things to records. 

This happens via  a [camel processor](http://camel.apache.org/processor.html) 

From there you would then turn the records in to ndarrays for dl4j-spark to consume.
