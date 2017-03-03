/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.camel.component;

import org.apache.camel.Exchange;
import org.apache.camel.impl.DefaultProducer;
import org.datavec.api.conf.Configuration;
import org.datavec.api.formats.input.InputFormat;
import org.datavec.api.io.WritableConverter;
import org.datavec.api.io.converters.SelfWritableConverter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;

import java.util.ArrayList;
import java.util.Collection;


/**
 * The DataVec producer.
 * Converts input records in to their final form
 * based on the input split generated from
 * the given exchange.
 *
 * @author Adam Gibson
 */
public class DataVecProducer extends DefaultProducer {
    private Class<? extends InputFormat> inputFormatClazz;
    private Class<? extends DataVecMarshaller> marshallerClazz;
    private InputFormat inputFormat;
    private Configuration configuration;
    private WritableConverter writableConverter;
    private DataVecMarshaller marshaller;


    public DataVecProducer(DataVecEndpoint endpoint) {
        super(endpoint);
        if (endpoint.getInputFormat() != null) {
            try {
                inputFormatClazz = (Class<? extends InputFormat>) Class.forName(endpoint.getInputFormat());
                inputFormat = inputFormatClazz.newInstance();
                marshallerClazz = (Class<? extends DataVecMarshaller>) Class.forName(endpoint.getInputMarshaller());
                Class<? extends WritableConverter> converterClazz =
                                (Class<? extends WritableConverter>) Class.forName(endpoint.getWritableConverter());
                writableConverter = converterClazz.newInstance();
                marshaller = marshallerClazz.newInstance();
                configuration = new Configuration();
                for (String prop : endpoint.getConsumerProperties().keySet())
                    configuration.set(prop, endpoint.getConsumerProperties().get(prop).toString());

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
        if (!(writableConverter instanceof SelfWritableConverter)) {
            newRecord = new ArrayList<>();
            while (reader.hasNext()) {
                Collection<Writable> newRecordAdd = new ArrayList<>();
                // create a message body
                Collection<Writable> next = reader.next();
                for (Writable writable : next) {
                    newRecordAdd.add(writableConverter.convert(writable));
                }


                newRecord.add(newRecordAdd);
            }
        } else {
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
