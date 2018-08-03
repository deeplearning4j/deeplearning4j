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

import lombok.Builder;
import org.apache.camel.CamelContext;

/**
 * Created by agibsonccc on 7/31/16.
 */
@Builder
public class NDArrayKafkaClient {
    private String kafkaUri;
    private String zooKeeperConnection;
    private CamelContext camelContext;
    private String kafkaTopic;
    public final static String NDARRAY_TYPE_HEADER = "ndarraytype";


    public NDArrayPublisher createPublisher() {
        return NDArrayPublisher.builder().kafkaUri(kafkaUri).topicName(kafkaTopic).camelContext(camelContext).build();
    }

    public NDArrayConsumer createConsumer() {
        return NDArrayConsumer.builder().camelContext(camelContext).kafkaUri(kafkaUri).topicName(kafkaTopic).build();
    }

}
