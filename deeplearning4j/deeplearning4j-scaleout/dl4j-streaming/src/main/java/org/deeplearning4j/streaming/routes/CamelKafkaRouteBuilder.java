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

package org.deeplearning4j.streaming.routes;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.camel.CamelContext;
import org.apache.camel.Processor;
import org.apache.camel.builder.RouteBuilder;

/**
 * A Camel Java DSL Router
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class CamelKafkaRouteBuilder extends RouteBuilder {
    private String topicName;
    private String kafkaBrokerList;
    private String writableConverter = "org.datavec.api.io.converters.SelfWritableConverter";
    private String datavecMarshaller = "org.datavec.camel.component.csv.marshaller.ListStringInputMarshaller";
    private String inputUri;
    private String inputFormat;
    private Processor processor;
    private String dataTypeUnMarshal;
    private String zooKeeperHost = "localhost";
    private int zooKeeperPort = 2181;

    /**
     * Let's configure the Camel routing rules using Java code...
     */
    @Override
    public void configure() {
        from(inputUri).unmarshal(dataTypeUnMarshal)
                        .to(String.format("datavec://%s?inputMarshaller=%s&writableConverter=%s", inputFormat,
                                        datavecMarshaller, writableConverter))
                        .process(processor).to(String.format("kafka:%s?topic=%s", kafkaBrokerList, topicName,
                                        zooKeeperHost, zooKeeperPort));
    }



    public void setContext(CamelContext camelContext) {
        super.setContext(camelContext);
    }



    public static class Builder {
        private String writableConverter = "org.datavec.api.io.converters.SelfWritableConverter";
        private String datavecMarshaller = "org.datavec.camel.component.csv.marshaller.ListStringInputMarshaller";
        private String inputUri;
        private String topicName;
        private String kafkaBrokerList = "localhost:9092";
        private CamelContext camelContext;
        private String inputFormat;
        private Processor processor;
        private String dataTypeUnMarshal;
        private String zooKeeperHost = "localhost";
        private int zooKeeperPort = 2181;

        public Builder zooKeeperHost(String zooKeeperHost) {
            this.zooKeeperHost = zooKeeperHost;
            return this;
        }

        public Builder zooKeeperPort(int zooKeeperPort) {
            this.zooKeeperPort = zooKeeperPort;
            return this;
        }

        public Builder processor(Processor processor) {
            this.processor = processor;
            return this;
        }

        public Builder kafkaBrokerList(String kafkaBrokerList) {
            this.kafkaBrokerList = kafkaBrokerList;
            return this;
        }

        public Builder inputFormat(String inputFormat) {
            this.inputFormat = inputFormat;
            return this;
        }

        public Builder camelContext(CamelContext camelContext) {
            this.camelContext = camelContext;
            return this;
        }

        public Builder inputUri(String inputUri) {
            this.inputUri = inputUri;
            return this;
        }

        public Builder writableConverter(String writableConverter) {
            this.writableConverter = writableConverter;
            return this;
        }


        public Builder datavecMarshaller(String datavecMarshaller) {
            this.datavecMarshaller = datavecMarshaller;
            return this;
        }

        public Builder dataTypeUnMarshal(String dataTypeUnMarshal) {
            this.dataTypeUnMarshal = dataTypeUnMarshal;
            return this;
        }


        public Builder topicName(String topicName) {
            this.topicName = topicName;
            return this;
        }

        private void assertStringNotNUllOrEmpty(String value, String name) {
            if (value == null || value.isEmpty())
                throw new IllegalStateException(String.format("Please define a %s", name));

        }

        public CamelKafkaRouteBuilder build() {
            CamelKafkaRouteBuilder routeBuilder;
            assertStringNotNUllOrEmpty(inputUri, "input uri");
            assertStringNotNUllOrEmpty(topicName, "topic name");
            assertStringNotNUllOrEmpty(kafkaBrokerList, "kafka broker");
            assertStringNotNUllOrEmpty(inputFormat, "input format");
            routeBuilder = new CamelKafkaRouteBuilder(topicName, kafkaBrokerList, writableConverter, datavecMarshaller,
                            inputUri, inputFormat, processor, dataTypeUnMarshal, zooKeeperHost, zooKeeperPort);
            if (camelContext != null)
                routeBuilder.setContext(camelContext);
            return routeBuilder;
        }

    }



}
