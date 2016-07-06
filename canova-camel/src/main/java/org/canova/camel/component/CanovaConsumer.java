package org.canova.camel.component;


import org.apache.camel.Exchange;
import org.apache.camel.Processor;
import org.apache.camel.impl.ScheduledPollConsumer;
import org.canova.api.conf.Configuration;
import org.canova.api.formats.input.InputFormat;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.InputSplit;

/**
 * The canova consumer.
 * @author Adam Gibson
 */
public class CanovaConsumer extends ScheduledPollConsumer {
    private final CanovaEndpoint endpoint;
    private Class<? extends InputFormat> inputFormatClazz;
    private Class<? extends CanovaMarshaller> marshallerClazz;
    private InputFormat inputFormat;
    private Configuration configuration;
    private CanovaMarshaller marshaller;


    public CanovaConsumer(CanovaEndpoint endpoint, Processor processor) {
        super(endpoint, processor);
        this.endpoint = endpoint;

        try {
            inputFormatClazz = (Class<? extends InputFormat>) Class.forName(endpoint.getInputFormat());
            inputFormat = inputFormatClazz.newInstance();
            marshallerClazz = (Class<? extends CanovaMarshaller>) Class.forName(endpoint.getInputMarshaller());
            marshaller = marshallerClazz.newInstance();
            configuration = new Configuration();
            for(String prop : endpoint.getConsumerProperties().keySet())
                configuration.set(prop,endpoint.getConsumerProperties().get(prop).toString());

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    //stub, still need to fill out more of the end point yet..endpoint will likely be initialized with a split
    protected InputSplit inputFromExchange(Exchange exchange) {
        return marshaller.getSplit(exchange);
    }

    @Override
    protected int poll() throws Exception {
        Exchange exchange = endpoint.createExchange();
        InputSplit split = inputFromExchange(exchange);
        RecordReader reader = inputFormat.createReader(split,configuration);
        int numMessagesPolled = 0;
        while(reader.hasNext()) {
            // create a message body
            while(reader.hasNext()) {
                exchange.getIn().setBody(reader.next());

                try {
                    // send message to next processor in the route
                    getProcessor().process(exchange);
                    numMessagesPolled++; // number of messages polled
                } finally {
                    // log exception if an exception occurred and was not handled
                    if (exchange.getException() != null) {
                        getExceptionHandler().handleException("Error processing exchange", exchange, exchange.getException());
                    }
                }
            }


        }

        return numMessagesPolled;
    }
}
