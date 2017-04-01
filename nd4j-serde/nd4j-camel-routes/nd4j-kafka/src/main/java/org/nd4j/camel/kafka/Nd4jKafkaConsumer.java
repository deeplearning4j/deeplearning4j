package org.nd4j.camel.kafka;

import lombok.AllArgsConstructor;
import lombok.Builder;
import org.apache.camel.CamelContext;
import org.apache.camel.ConsumerTemplate;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by agibsonccc on 7/19/16.
 */
@AllArgsConstructor
@Builder
public class Nd4jKafkaConsumer {
    private KafkaConnectionInformation connectionInformation;
    private ConsumerTemplate consumerTemplate;
    private CamelContext camelContext;

    /**
     * Receive an ndarray
     * @return
     */
    public INDArray receive() {
        if (consumerTemplate == null)
            consumerTemplate = camelContext.createConsumerTemplate();
        return consumerTemplate.receiveBody("direct:receive", INDArray.class);
    }

}
