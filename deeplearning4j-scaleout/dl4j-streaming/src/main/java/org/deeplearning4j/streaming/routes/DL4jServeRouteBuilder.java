package org.deeplearning4j.streaming.routes;

import lombok.AllArgsConstructor;
import lombok.Builder;
import org.apache.camel.Exchange;
import org.apache.camel.Processor;
import org.apache.camel.builder.RouteBuilder;
import org.apache.commons.net.util.Base64;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;

/**
 * Serve results from a kafka queue.
 * The input to the route can either be a pre serialized ndarray
 * or a normal ndarray itself.
 *
 * @author Adam Gibson
 */
@AllArgsConstructor
@Builder
public class DL4jServeRouteBuilder extends RouteBuilder {
    protected String modelUri;
    protected String kafkaBroker;
    protected String consumingTopic;
    protected boolean computationGraph;
    protected String outputUri;
    protected Processor finalProcessor;
    protected String groupId = "dl4j-serving";
    protected String zooKeeperHost = "localhost";
    protected int zooKeeperPort = 2181;
    //default no-op
    protected Processor beforeProcessor;


    /**
     * <b>Called on initialization to build the routes using the fluent builder syntax.</b>
     * <p/>
     * This is a central method for RouteBuilder implementations to implement
     * the routes using the Java fluent builder syntax.
     *
     * @throws Exception can be thrown during configuration
     */
    @Override
    public void configure() throws Exception {
        if (groupId == null)
            groupId = "dl4j-serving";
        if (zooKeeperHost == null)
            zooKeeperHost = "localhost";
        String kafkaUri = String.format("kafka:%s?topic=%s&groupId=%s", kafkaBroker, consumingTopic, groupId);
        if (beforeProcessor == null) {
            beforeProcessor = new Processor() {
                @Override
                public void process(Exchange exchange) throws Exception {

                }
            };
        }
        from(kafkaUri).process(beforeProcessor).process(new Processor() {
            @Override
            public void process(Exchange exchange) throws Exception {
                INDArray predict;
                if (exchange.getIn().getBody() instanceof byte[]) {
                    byte[] o = (byte[]) exchange.getIn().getBody();
                    byte[] arr = Base64.decodeBase64(new String(o));
                    ByteArrayInputStream bis = new ByteArrayInputStream(arr);
                    DataInputStream dis = new DataInputStream(bis);
                    predict = Nd4j.read(dis);
                } else
                    predict = (INDArray) exchange.getIn().getBody();

                if (computationGraph) {
                    ComputationGraph graph = ModelSerializer.restoreComputationGraph(modelUri);
                    INDArray[] output = graph.output(predict);
                    exchange.getOut().setBody(output);
                    exchange.getIn().setBody(output);

                } else {
                    MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(modelUri);
                    INDArray output = network.output(predict);
                    exchange.getOut().setBody(output);
                    exchange.getIn().setBody(output);
                }


            }
        }).process(finalProcessor).to(outputUri);
    }
}
