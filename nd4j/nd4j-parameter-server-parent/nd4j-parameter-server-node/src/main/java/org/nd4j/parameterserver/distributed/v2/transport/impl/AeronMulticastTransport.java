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

import io.aeron.FragmentAssembler;
import io.aeron.Publication;
import io.aeron.Subscription;
import io.aeron.logbuffer.Header;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.agrona.DirectBuffer;
import org.agrona.concurrent.SleepingIdleStrategy;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.enums.MeshBuildMode;
import org.nd4j.parameterserver.distributed.v2.enums.PropagationMode;
import org.nd4j.parameterserver.distributed.v2.enums.TransmissionStatus;
import org.nd4j.parameterserver.distributed.v2.messages.BroadcastableMessage;
import org.nd4j.parameterserver.distributed.v2.messages.INDArrayMessage;
import org.nd4j.parameterserver.distributed.v2.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.v2.util.MeshOrganizer;

import java.io.IOException;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;

/**
 * This class is a UDP Multicast implementation of Transport interface, based on Aeron
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class AeronMulticastTransport extends AeronUdpTransport {
    // non-null only on master
    private Publication multicastPublication;

    // non-null only on worker nodes
    private Subscription multicastSubscription;

    // this lock is used only for multicast stuff
    private ReentrantLock multicastLock = new ReentrantLock();

    protected volatile FragmentAssembler multicastMessageHandler;

    public AeronMulticastTransport(@NonNull String rootIp, int rootPort, @NonNull VoidConfiguration configuration) {
        this(rootIp, rootPort, rootIp, rootPort, configuration);
    }


    public AeronMulticastTransport(@NonNull String ownIp, int ownPort, @NonNull String rootIp, int rootPort, @NonNull VoidConfiguration configuration) {
        super(ownIp, ownPort, rootIp, rootPort, configuration);

        if (voidConfiguration.getMeshBuildMode() != MeshBuildMode.PLAIN) {
            throw new ND4JIllegalStateException("Multicast transport can only work in PLAIN MODE");
        }
    }

    @Override
    public String id() {
        return id;
    }

    protected String getMulticastChannelUri() {
        String multicastChannelUri = "aeron:udp?endpoint=" + voidConfiguration.getMulticastNetwork() + ":"
                + voidConfiguration.getMulticastPort();
        if (voidConfiguration.getMulticastInterface() != null && !voidConfiguration.getMulticastInterface().isEmpty())
            multicastChannelUri = multicastChannelUri + "|interface=" + voidConfiguration.getMulticastInterface();

        multicastChannelUri = multicastChannelUri + "|ttl=" + voidConfiguration.getTtl();
        return multicastChannelUri;
    }

    protected void createMulticastPublisher() {
        // here we create master's multicast stream
        val multicastChannelUri = getMulticastChannelUri();

        log.info("Creating multicast publication [{}]", multicastChannelUri);

        multicastPublication = aeron.addPublication(multicastChannelUri, voidConfiguration.getStreamId() + 1);
    }

    protected void createMulticastSubscription() {
        if (masterMode)
            return;

        // here we connect to master's multicast stream
        val multicastChannelUri = getMulticastChannelUri();

        log.info("Creating multicast subscription [{}]", multicastChannelUri);

        multicastMessageHandler = new FragmentAssembler((buffer, offset, length, header) -> multicastMessageHandler(buffer, offset, length, header));

        multicastSubscription = aeron.addSubscription(multicastChannelUri, voidConfiguration.getStreamId() + 1);

        // dedicated reader thread for multicast thread
        messagesExecutorService.execute(new Runnable() {
            @Override
            public void run() {
                val idler = new SleepingIdleStrategy(5000);
                while (true) {
                    idler.idle(multicastSubscription.poll(multicastMessageHandler, 1024));
                }
            }
        });
    }

    /**
     * This method converts aeron buffer into VoidMessage and puts into temp queue for further processing
     *
     * @param buffer
     * @param offset
     * @param length
     * @param header
     */
    protected void multicastMessageHandler(DirectBuffer buffer, int offset, int length, Header header) {
        byte[] data = new byte[length];
        buffer.getBytes(offset, data);

        // deserialize message
        val message = VoidMessage.fromBytes(data);

        log.info("Got [{}] message from multicast channel [{}]; aeronQueue size: [{}]; baseQueue size: [{}]", message.getClass().getSimpleName(), message.getOriginatorId(), aeronMessageQueue.size(), messageQueue.size());

        // we're just putting deserialized message into the buffer
        try {
            aeronMessageQueue.put(message);
        } catch (InterruptedException e) {
            // :(
            throw new RuntimeException(e);
        }
    }

    @Override
    public void propagateMessage(VoidMessage voidMessage, PropagationMode mode) throws IOException {
        if (voidMessage instanceof INDArrayMessage) {
            // since we're using multicast here, we don't want this message to be applied to original sender
            historyHolder.storeIfUnknownMessageId(voidMessage.getMessageId());
        }
        super.propagateMessage(voidMessage, mode);
    }

    @Override
    protected void createSubscription() {
        super.createSubscription();

        createMulticastSubscription();
    }

    protected void sendMulticastMessage(VoidMessage message) {
        log.info("Trying to send multicast message [{}]", message.getClass().getSimpleName());
        val buf = message.asUnsafeBuffer();
        TransmissionStatus status = TransmissionStatus.UNKNOWN;
        while (status != TransmissionStatus.OK) {
            try {
                multicastLock.lock();

                status = TransmissionStatus.fromLong(multicastPublication.offer(buf));
            } finally {
                multicastLock.unlock();
            }

            // sleep before retransmit
            if (status != TransmissionStatus.OK)
                LockSupport.parkNanos(50000);
        }

        log.info("Successfully sent multicast message [{}]", message.getClass().getSimpleName());
    }

    @Override
    protected void propagateBroadcastableMessage(BroadcastableMessage voidMessage, PropagationMode mode) {
        // in multicast transport we don't need propagation for worker nodes, since every online node gets the same data
        if (masterMode && voidMessage instanceof INDArrayMessage) {

            try {
                val splits = splitter.split(voidMessage, voidConfiguration.getMaxChunkSize());

                for(val m:splits) {
                    sendMulticastMessage(m);
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public void onMeshUpdate(MeshOrganizer mesh) {
        super.onMeshUpdate(mesh);
    }


    @Override
    public synchronized void launchAsMaster() {
        // we create multicast channel here first
        createMulticastPublisher();;

        super.launchAsMaster();
    }
}
