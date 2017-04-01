package org.nd4j.camel.kafka;

import lombok.AllArgsConstructor;
import lombok.Builder;
import org.apache.camel.CamelContext;
import org.apache.camel.ProducerTemplate;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by agibsonccc on 7/19/16.
 */
@AllArgsConstructor
@Builder
public class Nd4jKafkaProducer {

    private KafkaConnectionInformation connectionInformation;
    private CamelContext camelContext;
    private ProducerTemplate producerTemplate;

    /**
     * Publish to a kafka topic
     * based on the connection information
     * @param arr
     */
    public void publish(INDArray arr) {
        if (producerTemplate == null)
            producerTemplate = camelContext.createProducerTemplate();
        producerTemplate.sendBody("direct:start", arr);
    }


}
