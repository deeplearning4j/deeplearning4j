package org.deeplearning4j.streaming.pipeline.kafka;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.camel.CamelContext;
import org.apache.camel.Exchange;
import org.apache.camel.Processor;
import org.apache.camel.component.kafka.KafkaConstants;
import org.apache.camel.impl.DefaultCamelContext;
import org.apache.commons.net.util.Base64;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.streaming.routes.CamelKafkaRouteBuilder;
import org.deeplearning4j.streaming.serde.RecordSerializer;

import java.util.Collection;
import java.util.Random;
import java.util.UUID;


/**
 * A base kafka pieline that handles
 * connecting to kafka and consuming from a stream.
 *
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor
public abstract class BaseKafkaPipeline<E,RECORD_CONVERTER_FUNCTION> {

    protected String kafkaTopic;
    protected String inputUri;
    protected String inputFormat;
    protected String kafkaBroker;
    protected String zkHost;
    protected CamelContext camelContext;
    protected String hadoopHome;
    protected String dataType;
    protected String sparkAppName = "datavec";
    protected int kafkaPartitions = 1;
    protected RECORD_CONVERTER_FUNCTION recordToDataSetFunction;
    protected int numLabels;
    protected E dataset;



    /**
     * Initialize the pipeline
     * setting up camel routes,
     * kafka,datavec,and the
     * spark streaming DAG.
     * @throws Exception
     */
    public    void init() throws Exception {
        if (camelContext == null)
            camelContext = new DefaultCamelContext();

        camelContext.addRoutes(new CamelKafkaRouteBuilder.Builder().
                camelContext(camelContext)
                .inputFormat(inputFormat).
                        topicName(kafkaTopic).camelContext(camelContext)
                .dataTypeUnMarshal(dataType)
                .inputUri(inputUri).
                        kafkaBrokerList(kafkaBroker).processor(new Processor() {
                    @Override
                    public void process(Exchange exchange) throws Exception {
                        Collection<Collection<Writable>> record = (Collection<Collection<Writable>>) exchange.getIn().getBody();
                        exchange.getIn().setHeader(KafkaConstants.KEY, UUID.randomUUID().toString());
                        exchange.getIn().setHeader(KafkaConstants.PARTITION_KEY, new Random().nextInt(kafkaPartitions));
                        byte[] bytes = new RecordSerializer().serialize(kafkaTopic, record);
                        String base64 = Base64.encodeBase64String(bytes);
                        exchange.getIn().setBody(base64, String.class);
                    }
                }).build());


        if(hadoopHome == null)
            hadoopHome = System.getProperty("java.io.tmpdir");
        System.setProperty("hadoop.home.dir", hadoopHome);

        initComponents();
    }


    /**
     * Start the camel context
     * used for etl
     * @throws Exception
     */
    public void startCamel() throws Exception {
        camelContext.start();
    }

    /**
     * Stop the camel context
     * used for etl
     * @throws Exception
     */
    public void stopCamel() throws Exception {
        camelContext.stop();
    }

    /**
     * Initialize implementation specific components
     */
    public abstract void initComponents();


    /**
     * Create the streaming result
     * @return the stream
     */
    public  abstract E createStream();

    /**
     * Starts the streaming consumption
     */
    public  void startStreamingConsumption() {
        startStreamingConsumption(-1);
    }

    /**
     * Starts the streaming consumption
     * @param timeout how long to run consumption for (-1 for infinite)
     */
    public abstract void startStreamingConsumption(long timeout);

    /**
     * Run the pipeline
     * @throws Exception
     */
    public E run() throws Exception {
        // Start the computation
        startCamel();
        dataset = createStream();
        stopCamel();
        return dataset;
    }


}
