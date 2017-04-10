package org.deeplearning4j.streaming.kafka;

import lombok.Builder;
import lombok.Data;

/**
 * Kafka uri builder
 *
 * @author Adam Gibson
 */
@Data
@Builder
public class KafkaUriBuilder {
    private String kafkaBroker;
    private String consumingTopic;
    private String groupId;
    private String zooKeeperHost;
    private int zooKeeperPort;

    public String uri() {
        return String.format("kafka://%s?topic=%s&groupId=%s", kafkaBroker, consumingTopic, groupId);
    }
}
