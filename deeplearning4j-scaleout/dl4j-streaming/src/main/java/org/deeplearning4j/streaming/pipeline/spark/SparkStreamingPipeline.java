package org.deeplearning4j.streaming.pipeline.spark;

import lombok.Builder;
import lombok.Data;
import org.apache.camel.CamelContext;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.streaming.Duration;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaDStream;
import org.apache.spark.streaming.api.java.JavaPairInputDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import org.apache.spark.streaming.kafka.KafkaUtils;
import org.deeplearning4j.streaming.conversion.dataset.RecordToDataSet;
import org.deeplearning4j.streaming.pipeline.kafka.BaseKafkaPipeline;
import org.nd4j.linalg.dataset.DataSet;

import java.util.Collections;

/**
 * Spark streaming pipeline.
 *
 * @author Adam Gibson
 */
@Data
public class SparkStreamingPipeline extends BaseKafkaPipeline<JavaDStream<DataSet>,RecordToDataSet> {
    protected JavaStreamingContext jssc;
    protected SparkConf sparkConf;
    protected Function<JavaPairRDD<String, String>, Void> streamProcessor;
    protected Duration streamingDuration =  Durations.seconds(1);
    protected String sparkMaster;
    protected Function<JavaRDD<DataSet>, Void> datasetConsumption;

    @Builder
    public SparkStreamingPipeline(String kafkaTopic, String inputUri, String inputFormat, String kafkaBroker, String zkHost, CamelContext camelContext, String hadoopHome, String dataType, String sparkAppName, int kafkaPartitions, RecordToDataSet recordToDataSetFunction, int numLabels, JavaDStream<DataSet> dataset, JavaStreamingContext jssc, SparkConf sparkConf, Function<JavaPairRDD<String, String>, Void> streamProcessor, Duration streamingDuration, String sparkMaster) {
        super(kafkaTopic, inputUri, inputFormat, kafkaBroker, zkHost, camelContext, hadoopHome, dataType, sparkAppName, kafkaPartitions, recordToDataSetFunction, numLabels, dataset);
        this.jssc = jssc;
        this.sparkConf = sparkConf;
        this.streamProcessor = streamProcessor;
        this.streamingDuration = streamingDuration;
        this.sparkMaster = sparkMaster;
    }

    @Override
    public void initComponents() {
        sparkConf = new SparkConf().setAppName(sparkAppName).setMaster(sparkMaster);
        jssc = new JavaStreamingContext(sparkConf, streamingDuration);
    }

    /**
     * Create the streaming result
     *
     * @return the stream
     */
    @Override
    public JavaDStream<DataSet> createStream() {
        JavaPairInputDStream<String, String> messages = KafkaUtils.createStream(
                jssc,
                zkHost,
                "datavec",
                Collections.singletonMap(kafkaTopic, kafkaPartitions));
        JavaDStream<DataSet> dataset = messages.flatMap(new DataSetFlatmap(numLabels,recordToDataSetFunction)).cache();
        return dataset;
    }

    /**
     * Starts the streaming consumption
     */
    @Override
    public void startStreamingConsumption(long timeout) {
        jssc.start();
        if(timeout < 0)
            jssc.awaitTermination();
        else
            jssc.awaitTermination(timeout);
    }
}
