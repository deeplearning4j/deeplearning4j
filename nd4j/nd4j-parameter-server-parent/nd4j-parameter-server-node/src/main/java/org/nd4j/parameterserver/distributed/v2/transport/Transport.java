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

package org.nd4j.parameterserver.distributed.v2.transport;

import io.reactivex.functions.Consumer;
import org.nd4j.parameterserver.distributed.v2.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.v2.messages.INDArrayMessage;
import org.reactivestreams.Publisher;

import java.io.IOException;

public interface Transport {

    /**
     * This method returns consumer that accepts messages for delivery
     * @return
     */
    Consumer<VoidMessage> outgoingConsumer();

    /**
     * This method returns flow of messages for parameter server
     * @return
     */
    Publisher<INDArrayMessage> incomingPublisher();

    /**
     * This method starts  this Transport instance
     */
    void launch();

    /**
     * This method will send message to the network, using tree structure
     * @param message
     */
    void propagateMessage(VoidMessage message) throws IOException;

    /**
     * This method will send message to the node specified by Id
     *
     * @param message
     * @param id
     */
    void sendMessage(VoidMessage message, String id);

    /**
     * This method will be invoked for all incoming messages
     * PLEASE NOTE: this method is mostly suited for tests
     *
     * @param message
     */
    void processMessage(VoidMessage message);
}
