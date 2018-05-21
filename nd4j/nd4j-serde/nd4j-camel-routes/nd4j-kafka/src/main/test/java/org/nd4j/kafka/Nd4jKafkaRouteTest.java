package org.nd4j.kafka;

import org.apache.camel.CamelContext;
import org.apache.camel.impl.DefaultCamelContext;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.camel.kafka.KafkaConnectionInformation;
import org.nd4j.camel.kafka.Nd4jKafkaConsumer;
import org.nd4j.camel.kafka.Nd4jKafkaProducer;
import org.nd4j.camel.kafka.Nd4jKafkaRoute;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 7/19/16.
 */
public class Nd4jKafkaRouteTest {
    private EmbeddedKafkaCluster kafka;
    private EmbeddedZookeeper zk;
    private CamelContext camelContext;
    public final static String TOPIC = "nd4jtest";
    public final static String GROUP_ID = "nd4j";
    private KafkaConnectionInformation connectionInformation;



    @Before
    public void before() throws Exception {
        zk = new EmbeddedZookeeper(TestUtils.getAvailablePort());
        zk.startup();
        kafka = new EmbeddedKafkaCluster(zk.getConnection());
        kafka.startup();
        kafka.createTopics(TOPIC);
        camelContext = new DefaultCamelContext();
        camelContext.start();
        connectionInformation = KafkaConnectionInformation.builder().groupId(GROUP_ID).topicName(TOPIC)
                        .zookeeperHost("localhost").zookeeperPort(zk.getPort()).kafkaBrokerList(kafka.getBrokerList())
                        .build();
        camelContext.addRoutes(Nd4jKafkaRoute.builder().kafkaConnectionInformation(connectionInformation).build());
    }

    @After
    public void after() throws Exception {
        if (kafka != null)
            kafka.shutdown();
        if (zk != null)
            zk.shutdown();
        if (camelContext != null)
            camelContext.stop();
    }


    @Test
    public void testKafkaRoute() throws Exception {
        Nd4jKafkaProducer kafkaProducer = Nd4jKafkaProducer.builder().camelContext(camelContext)
                        .connectionInformation(connectionInformation).build();
        kafkaProducer.publish(Nd4j.create(4));
        Nd4jKafkaConsumer consumer = Nd4jKafkaConsumer.builder().camelContext(camelContext)
                        .connectionInformation(connectionInformation).build();
        assertEquals(Nd4j.create(4), consumer.receive());
    }


}
