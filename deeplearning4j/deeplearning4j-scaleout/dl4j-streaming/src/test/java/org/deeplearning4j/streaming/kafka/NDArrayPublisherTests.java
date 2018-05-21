package org.deeplearning4j.streaming.kafka;

import org.apache.camel.CamelContext;
import org.apache.camel.impl.DefaultCamelContext;
import org.deeplearning4j.streaming.embedded.EmbeddedKafkaCluster;
import org.deeplearning4j.streaming.embedded.EmbeddedZookeeper;
import org.deeplearning4j.streaming.embedded.TestUtils;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 7/30/16.
 */
public class NDArrayPublisherTests {


    @Test
    public void testPublish() throws Exception {
        String topicName = "testkafka";
        EmbeddedZookeeper embeddedZookeeper = new EmbeddedZookeeper(TestUtils.getAvailablePort());
        embeddedZookeeper.startup();
        EmbeddedKafkaCluster embeddedKafkaCluster = new EmbeddedKafkaCluster(embeddedZookeeper.getConnection());
        embeddedKafkaCluster.startup();
        embeddedKafkaCluster.createTopics(topicName);
        CamelContext camelContext = new DefaultCamelContext();
        INDArray arr = Nd4j.linspace(1, 4, 4);
        String kafkaUri = KafkaUriBuilder.builder().kafkaBroker(embeddedKafkaCluster.getBrokerList())
                        .consumingTopic(topicName).groupId("dl4jgroup").zooKeeperHost("localhost")
                        .zooKeeperPort(embeddedZookeeper.getPort()).build().uri();
        NDArrayPublisher publisher = NDArrayPublisher.builder().camelContext(camelContext).kafkaUri(kafkaUri)
                        .topicName(topicName).build();
        camelContext.start();
        publisher.start();

        NDArrayConsumer consumer = NDArrayConsumer.builder().kafkaUri(kafkaUri).topicName(topicName)
                        .camelContext(camelContext).build();
        consumer.start();


        publisher.publish(arr);

        Thread.sleep(5000);

        INDArray get = consumer.getINDArray();
        assertEquals(arr, get);


        embeddedKafkaCluster.shutdown();
        embeddedZookeeper.shutdown();
        camelContext.stop();
    }

}
