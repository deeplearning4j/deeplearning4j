package org.nd4j.camel.kafka;

import kafka.serializer.StringEncoder;
import lombok.Builder;
import lombok.Data;

import java.io.Serializable;

/**
 * Kafka connection information
 * to generate camel uris
 *
 * @author Adam Gibson
 */
@Builder
@Data
public class KafkaConnectionInformation implements Serializable {
    private String zookeeperHost;
    private int zookeeperPort;
    private String kafkaBrokerList;
    private String topicName;
    private String groupId;

    /**
     * Returns a kafka connection uri
     * @return a kafka connection uri
     * represented by this connection information
     */
    public String kafkaUri() {
        return String.format(
                        "kafka://%s?topic=%s&groupId=%s&zookeeperHost=%s&zookeeperPort=%d&serializerClass=%s&keySerializerClass=%s",
                        kafkaBrokerList, topicName, groupId, zookeeperHost, zookeeperPort,
                        StringEncoder.class.getName(), StringEncoder.class.getName());
    }
}
