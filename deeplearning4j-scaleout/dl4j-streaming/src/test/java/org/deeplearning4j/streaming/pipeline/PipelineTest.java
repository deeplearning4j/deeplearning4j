package org.deeplearning4j.streaming.pipeline;


import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaDStream;
import org.deeplearning4j.streaming.conversion.dataset.CSVRecordToDataSet;
import org.deeplearning4j.streaming.embedded.EmbeddedKafkaCluster;
import org.deeplearning4j.streaming.embedded.EmbeddedZookeeper;
import org.deeplearning4j.streaming.embedded.TestUtils;
import org.deeplearning4j.streaming.pipeline.spark.PrintDataSet;
import org.deeplearning4j.streaming.pipeline.spark.SparkStreamingPipeline;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by agibsonccc on 6/10/16.
 */
public class PipelineTest {
    private static  EmbeddedZookeeper zookeeper;
    private static EmbeddedKafkaCluster kafkaCluster;
    private static int zkPort;
    public final static String LOCALHOST = "localhost";

    @BeforeClass
    public static void init() throws Exception {
        zkPort = TestUtils.getAvailablePort();
        zookeeper = new EmbeddedZookeeper(zkPort);
        zookeeper.startup();
        kafkaCluster = new EmbeddedKafkaCluster(LOCALHOST + ":" + zkPort);
        kafkaCluster.startup();
    }

    @AfterClass
    public static void after() {
        kafkaCluster.shutdown();
        zookeeper.shutdown();
    }


    @Test
    public void testPipeline() throws Exception {
        SparkStreamingPipeline pipeline = SparkStreamingPipeline.builder()
                .dataType("csv").kafkaBroker(kafkaCluster.getBrokerList())
                .inputFormat("org.datavec.api.formats.input.impl.ListStringInputFormat")
                .inputUri("file:src/test/resources/?fileName=iris.dat&noop=true").streamingDuration(Durations.seconds(1))
                .kafkaPartitions(1).kafkaTopic("test3").sparkMaster("local[*]").numLabels(3).recordToDataSetFunction(new CSVRecordToDataSet())
                .zkHost("localhost:" + zkPort).sparkAppName("datavec").build();
        pipeline.init();

        final JavaDStream<DataSet> dataSetJavaDStream =  pipeline.run();
         //NOTE THAT YOU NEED TO DO SOMETHING WITH THE STREAM OTHERWISE IT ERRORS OUT.
        //ALSO NOTE HERE THAT YOU NEED TO HAVE THE FUNCTION BE AN OBJECT NOT AN ANONYMOUS
        //CLASS BECAUSE OF TASK SERIALIZATION
        dataSetJavaDStream.foreach(new PrintDataSet());

        pipeline.startStreamingConsumption(1000);

    }


}
