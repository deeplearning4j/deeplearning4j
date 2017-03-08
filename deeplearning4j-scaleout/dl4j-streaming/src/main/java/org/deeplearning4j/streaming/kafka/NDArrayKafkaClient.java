package org.deeplearning4j.streaming.kafka;

import lombok.Builder;
import org.apache.camel.CamelContext;

/**
 * Created by agibsonccc on 7/31/16.
 */
@Builder
public class NDArrayKafkaClient {
    private String kafkaUri;
    private String zooKeeperConnection;
    private CamelContext camelContext;
    private String kafkaTopic;
    public final static String NDARRAY_TYPE_HEADER = "ndarraytype";


    public NDArrayPublisher createPublisher() {
        return NDArrayPublisher.builder().kafkaUri(kafkaUri).topicName(kafkaTopic).camelContext(camelContext).build();
    }

    public NDArrayConsumer createConsumer() {
        return NDArrayConsumer.builder().camelContext(camelContext).kafkaUri(kafkaUri).topicName(kafkaTopic).build();
    }

}
