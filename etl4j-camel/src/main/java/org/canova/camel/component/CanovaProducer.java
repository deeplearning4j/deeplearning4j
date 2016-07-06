package org.canova.camel.component;

import org.apache.camel.Exchange;
import org.apache.camel.impl.DefaultProducer;
import org.canova.api.conf.Configuration;
import org.canova.api.formats.input.InputFormat;
import org.canova.api.io.WritableConverter;
import org.canova.api.io.converters.SelfWritableConverter;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;

import java.util.ArrayList;
import java.util.Collection;


/**
 * The canova producer.
 * Converts input records in to their final form
 * based on the input split generated from
 * the given exchange.
 *
 * @author Adam Gibson
 */
public class CanovaProducer extends DefaultProducer {
    private Class<? extends InputFormat> inputFormatClazz;
    private Class<? extends CanovaMarshaller> marshallerClazz;
    private InputFormat inputFormat;
    private Configuration configuration;
    private WritableConverter writableConverter;
    private CanovaMarshaller marshaller;


    public CanovaProducer(CanovaEndpoint endpoint) {
        super(endpoint);
        if(endpoint.getInputFormat() != null) {
            try {
                inputFormatClazz = (Class<? extends InputFormat>) Class.forName(endpoint.getInputFormat());
                inputFormat = inputFormatClazz.newInstance();
                marshallerClazz = (Class<? extends CanovaMarshaller>) Class.forName(endpoint.getInputMarshaller());
                Class<? extends WritableConverter> converterClazz = (Class<? extends WritableConverter>) Class.forName(endpoint.getWritableConverter());
                writableConverter = converterClazz.newInstance();
                marshaller = marshallerClazz.newInstance();
                configuration = new Configuration();
                for(String prop : endpoint.getConsumerProperties().keySet())
                    configuration.set(prop,endpoint.getConsumerProperties().get(prop).toString());

            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

    }


    //stub, still need to fill out more of the end point yet..endpoint will likely be initialized with a split
    protected InputSplit inputFromExchange(Exchange exchange) {
        return marshaller.getSplit(exchange);
    }


    @Override
    public void process(Exchange exchange) throws Exception {
        InputSplit split = inputFromExchange(exchange);
        RecordReader reader = inputFormat.createReader(split, configuration);
        Collection<Collection<Writable>> newRecord = new ArrayList<>();
        if(!(writableConverter instanceof SelfWritableConverter)) {
            newRecord = new ArrayList<>();
            while (reader.hasNext()) {
                Collection<Writable> newRecordAdd = new ArrayList<>();
                // create a message body
                Collection<Writable> next = reader.next();
                for(Writable writable : next) {
                    newRecordAdd.add(writableConverter.convert(writable));
                }


                newRecord.add(newRecordAdd);
            }
        }
        else {
            while (reader.hasNext()) {
                // create a message body
                Collection<Writable> next = reader.next();
                newRecord.add(next);
            }
        }


        exchange.getIn().setBody(newRecord);
        exchange.getOut().setBody(newRecord);
    }
}
