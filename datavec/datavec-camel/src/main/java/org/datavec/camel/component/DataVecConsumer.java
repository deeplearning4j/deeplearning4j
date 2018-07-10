/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.camel.component;


import org.apache.camel.Exchange;
import org.apache.camel.Processor;
import org.apache.camel.impl.ScheduledPollConsumer;
import org.datavec.api.conf.Configuration;
import org.datavec.api.formats.input.InputFormat;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.InputSplit;

/**
 * The DataVec consumer.
 * @author Adam Gibson
 */
public class DataVecConsumer extends ScheduledPollConsumer {
    private final DataVecEndpoint endpoint;
    private Class<? extends InputFormat> inputFormatClazz;
    private Class<? extends DataVecMarshaller> marshallerClazz;
    private InputFormat inputFormat;
    private Configuration configuration;
    private DataVecMarshaller marshaller;


    public DataVecConsumer(DataVecEndpoint endpoint, Processor processor) {
        super(endpoint, processor);
        this.endpoint = endpoint;

        try {
            inputFormatClazz = (Class<? extends InputFormat>) Class.forName(endpoint.getInputFormat());
            inputFormat = inputFormatClazz.newInstance();
            marshallerClazz = (Class<? extends DataVecMarshaller>) Class.forName(endpoint.getInputMarshaller());
            marshaller = marshallerClazz.newInstance();
            configuration = new Configuration();
            for (String prop : endpoint.getConsumerProperties().keySet())
                configuration.set(prop, endpoint.getConsumerProperties().get(prop).toString());

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
        RecordReader reader = inputFormat.createReader(split, configuration);
        int numMessagesPolled = 0;
        while (reader.hasNext()) {
            // create a message body
            while (reader.hasNext()) {
                exchange.getIn().setBody(reader.next());

                try {
                    // send message to next processor in the route
                    getProcessor().process(exchange);
                    numMessagesPolled++; // number of messages polled
                } finally {
                    // log exception if an exception occurred and was not handled
                    if (exchange.getException() != null) {
                        getExceptionHandler().handleException("Error processing exchange", exchange,
                                        exchange.getException());
                    }
                }
            }


        }

        return numMessagesPolled;
    }
}
