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

package org.nd4j.parameterserver.distributed.v2.transport.impl;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.v2.messages.RequestMessage;
import org.nd4j.parameterserver.distributed.v2.messages.VoidMessage;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * This class is a UDP implementation of Transport interface, based on Aeron
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class AeronUdpTransport extends BaseTransport {
    // this map holds outgoing connections, basically
    private Map<String, Object> remoteConnections = new ConcurrentHashMap<>();

    @Override
    public synchronized void launch() {
        // we set up aeron first

        super.launch();
    }

    @Override
    public synchronized void launchAsMaster() {
        // connection goes first

        super.launchAsMaster();
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public void sendMessage(VoidMessage message, String id) {
        if (message.getOriginatorId() == null)
            message.setOriginatorId(this.id());

        // TODO: get rid of UUID!!!11
        if (message instanceof RequestMessage) {
            if (((RequestMessage) message).getRequestId() == null)
                ((RequestMessage) message).setRequestId(java.util.UUID.randomUUID().toString());
        }
    }
}
