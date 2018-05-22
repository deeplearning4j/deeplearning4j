package org.deeplearning4j.streaming.kafka;

import lombok.AllArgsConstructor;
import lombok.Builder;
import org.apache.camel.CamelContext;
import org.apache.camel.ConsumerTemplate;
import org.apache.camel.Exchange;
import org.apache.camel.Processor;
import org.apache.camel.builder.RouteBuilder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.base64.Nd4jBase64;

/**
 * NDArray consumer for receiving
 * ndarrays off of kafka
 *
 * @author Adam Gibson
 */
@Builder
@AllArgsConstructor
public class NDArrayConsumer {
    private CamelContext camelContext;
    private ConsumerTemplate consumerTemplate;
    private String topicName;
    public final static String DIRECT_ROUTE = "seda:receive";
    private String kafkaUri;
    private boolean started = false;


    /**
     * Start the consumer
     * @throws Exception
     */
    public void start() throws Exception {
        if (started)
            return;

        camelContext.addRoutes(new RouteBuilder() {
            @Override
            public void configure() throws Exception {
                from(kafkaUri).process(new Processor() {
                    @Override
                    public void process(Exchange exchange) throws Exception {
                        byte[] message = (byte[]) exchange.getIn().getBody();
                        String base64 = new String(message);
                        if (!Nd4jBase64.isMultiple(base64)) {
                            INDArray get = Nd4jBase64.fromBase64(base64);
                            exchange.getIn().setBody(get);
                        } else {
                            INDArray[] arrs = Nd4jBase64.arraysFromBase64(exchange.getIn().getBody().toString());
                            exchange.getIn().setBody(arrs);
                        }
                    }
                }).to(DIRECT_ROUTE);
            }
        });

        if (consumerTemplate == null)
            consumerTemplate = camelContext.createConsumerTemplate();

    }


    /**
     * Receive an ndarray from the queue
     * @return the ndarray to get
     * @throws Exception
     */
    public INDArray[] getArrays() throws Exception {
        if (!started) {
            start();
            started = true;
        }
        return consumerTemplate.receiveBody(DIRECT_ROUTE, INDArray[].class);

    }


    public INDArray getINDArray() throws Exception {
        if (!started) {
            start();
            started = true;
        }
        return consumerTemplate.receiveBody(DIRECT_ROUTE, INDArray.class);
    }
}
