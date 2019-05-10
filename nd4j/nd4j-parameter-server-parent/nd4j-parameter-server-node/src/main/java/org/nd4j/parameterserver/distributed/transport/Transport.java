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

package org.nd4j.parameterserver.distributed.transport;

import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.messages.MeaningfulMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;

/**
 * Transport interface describes Client -> Shard, Shard -> Shard, Shard -> Client communication
 *
 * @author raver119@gmail.com
 */
@Deprecated
public interface Transport {
    enum ThreadingModel {
        SAME_THREAD, // DO NOT USE IT IN REAL ENVIRONMENT!!!11oneoneeleven
        SINGLE_THREAD, DEDICATED_THREADS,
    }

    void setIpAndPort(String ip, int port);

    String getIp();

    int getPort();

    short getShardIndex();


    short getTargetIndex();


    void addClient(String ip, int port);


    void addShard(String ip, int port);

    /**
     * This method does initialization of Transport instance
     *
     * @param voidConfiguration
     * @param role
     * @param localIp
     */
    void init(VoidConfiguration voidConfiguration, Clipboard clipboard, NodeRole role, String localIp, int localPort,
                    short shardIndex);


    /**
     * This method accepts message for delivery, routing is applied according on message opType
     *
     * @param message
     */
    void sendMessage(VoidMessage message);

    /**
     * This method accepts message for delivery, and blocks until response delivered
     *
     * @return
     */
    MeaningfulMessage sendMessageAndGetResponse(VoidMessage message);

    /**
     *
     * @param message
     */
    void sendMessageToAllShards(VoidMessage message);

    /**
     *
     * @param message
     */
    void sendMessageToAllClients(VoidMessage message, Long... exclusions);

    /**
     * This method accepts message from network
     *
     * @param message
     */
    void receiveMessage(VoidMessage message);

    /**
     * This method takes 1 message from "incoming messages" queue, blocking if queue is empty
     *
     * @return
     */
    VoidMessage takeMessage();

    /**
     * This method puts message into processing queue
     *
     * @param message
     */
    void putMessage(VoidMessage message);

    /**
     * This method peeks 1 message from "incoming messages" queue, returning null if queue is empty
     *
     * PLEASE NOTE: This method is suitable for debug purposes only
     *
     * @return
     */
    VoidMessage peekMessage();

    /**
     * This method starts transport mechanisms.
     *
     * PLEASE NOTE: init() method should be called prior to launch() call
     */
    void launch(ThreadingModel threading);

    /**
     * This method stops transport system.
     */
    void shutdown();

    /**
     * This method returns number of known Clients
     * @return
     */
    int numberOfKnownClients();

    /**
     * This method returns number of known Shards
     * @return
     */
    int numberOfKnownShards();

    /**
     * This method returns ID of this Transport instance
     * @return
     */
    long getOwnOriginatorId();
}
