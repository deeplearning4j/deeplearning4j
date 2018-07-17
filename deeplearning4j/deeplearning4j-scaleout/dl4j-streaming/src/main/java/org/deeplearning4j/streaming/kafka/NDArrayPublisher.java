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

package org.deeplearning4j.streaming.kafka;

import lombok.AllArgsConstructor;
import lombok.Builder;
import org.apache.camel.CamelContext;
import org.apache.camel.Exchange;
import org.apache.camel.Processor;
import org.apache.camel.ProducerTemplate;
import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.component.kafka.KafkaConstants;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.base64.Nd4jBase64;

import static org.deeplearning4j.streaming.kafka.NDArrayKafkaClient.NDARRAY_TYPE_HEADER;

/**
 * Send an ndarray to a kafka topic
 *
 * @author Adam Gibson
 */
@Builder
@AllArgsConstructor
public class NDArrayPublisher {
    private CamelContext camelContext;
    private String topicName;
    private String kafkaUri;
    private ProducerTemplate producerTemplate;
    private boolean started = false;
    public final static String DIRECT_ROUTE = "direct:send";

    public final static String NDARRAY_TYPE = "ndarraytype";

    /**
     * Publish an ndarray
     * @param arr the ndarray to publish
     */
    public void publish(INDArray[] arr) throws Exception {
        if (!started) {
            start();
        }
        producerTemplate.sendBody(DIRECT_ROUTE, arr);
    }

    /**
     * Publish an ndarray
     * @param arr the ndarray to publish
     */
    public void publish(INDArray arr) throws Exception {
        if (!started) {
            start();
        }
        producerTemplate.sendBody(DIRECT_ROUTE, arr);
    }

    /**
     * Start the publisher
     * @throws Exception
     */
    public void start() throws Exception {
        if (started)
            return;
        started = true;

        camelContext.addRoutes(new RouteBuilder() {
            @Override
            public void configure() throws Exception {
                from(DIRECT_ROUTE).process(new Processor() {
                    @Override
                    public void process(Exchange exchange) throws Exception {
                        Object body = exchange.getIn().getBody();
                        if (body instanceof INDArray) {
                            INDArray arr = (INDArray) body;
                            String arrBase = Nd4jBase64.base64String(arr);
                            exchange.getIn().setBody(arrBase);
                            exchange.getIn().setHeader(NDARRAY_TYPE_HEADER, NDArrayType.SINGLE.toString());
                        } else if (body instanceof INDArray[]) {
                            INDArray[] arrs = (INDArray[]) body;
                            String arrBase = Nd4jBase64.arraysToBase64(arrs);
                            exchange.getIn().setBody(arrBase);
                            exchange.getIn().setHeader(NDARRAY_TYPE_HEADER, NDArrayType.MULTI.toString());
                        }

                        exchange.getIn().setHeader(KafkaConstants.PARTITION_KEY, 0);
                        exchange.getIn().setHeader(KafkaConstants.KEY, "1");
                    }
                }).to(kafkaUri);
            }
        });

        if (producerTemplate == null)
            producerTemplate = camelContext.createProducerTemplate();
    }

}
