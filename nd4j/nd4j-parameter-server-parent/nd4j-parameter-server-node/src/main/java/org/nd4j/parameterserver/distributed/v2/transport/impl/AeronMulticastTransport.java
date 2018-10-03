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

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.v2.util.MeshOrganizer;

/**
 * This class is a UDP Multicast implementation of Transport interface, based on Aeron
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class AeronMulticastTransport extends AeronUdpTransport {
    public AeronMulticastTransport(@NonNull String rootIp, int rootPort, @NonNull VoidConfiguration configuration) {
        this(rootIp, rootPort, rootIp, rootPort, configuration);
    }


    public AeronMulticastTransport(@NonNull String ownIp, int ownPort, @NonNull String rootIp, int rootPort, @NonNull VoidConfiguration configuration) {
        super(ownIp, ownPort, rootIp, rootPort, configuration);
    }

    @Override
    public String id() {
        return id;
    }

    protected void createMulticastSubscription() {
        // here we connect to master's multicast stream
    }

    @Override
    protected void createSubscription() {
        // setting up multicast before other connections
        createMulticastSubscription();

        super.createSubscription();
    }

    @Override
    public void sendMessage(VoidMessage message, String id) {
        // we send server-related messages via unicast messages to master

        // training-related messages are sent to
    }

    @Override
    public void onMeshUpdate(MeshOrganizer mesh) {
        // if there's yet unknown nodes in mesh - connect to their multicast channels

        super.onMeshUpdate(mesh);
    }

    @Override
    public synchronized void launch() {
        // we create multicast channel here first

        super.launch();
    }

    @Override
    public synchronized void launchAsMaster() {
        // we create multicast channel here first

        super.launchAsMaster();
    }
}
