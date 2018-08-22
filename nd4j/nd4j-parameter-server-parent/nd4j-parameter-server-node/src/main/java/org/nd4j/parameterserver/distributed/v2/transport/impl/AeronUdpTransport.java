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

import io.aeron.Aeron;
import io.aeron.FragmentAssembler;
import io.aeron.Publication;
import io.aeron.Subscription;
import io.aeron.driver.MediaDriver;
import io.aeron.logbuffer.Header;
import lombok.Builder;
import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.agrona.DirectBuffer;
import org.agrona.concurrent.SleepingIdleStrategy;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.HashUtil;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.messages.RequestMessage;
import org.nd4j.parameterserver.distributed.v2.messages.VoidMessage;

import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * This class is a UDP implementation of Transport interface, based on Aeron
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class AeronUdpTransport extends BaseTransport {
    // this map holds outgoing connections, basically
    private Map<String, RemoteConnection> remoteConnections = new ConcurrentHashMap<>();

    protected final int MESSAGE_THREADS = 2;
    protected final int SUBSCRIPTION_THREADS = 1;

    protected Aeron aeron;
    protected Aeron.Context context;

    protected VoidConfiguration voidConfiguration;

    protected Subscription ownSubscription;
    protected FragmentAssembler messageHandler;
    protected Thread subscriptionThread;

    // TODO: move this to singleton holder
    protected MediaDriver driver;

    // this is intermediate buffer for incoming messages
    protected BlockingQueue<VoidMessage> messageQueue = new LinkedTransferQueue<>();

    // this executor service han
    protected ExecutorService messagesExecutorService = Executors.newFixedThreadPool(MESSAGE_THREADS + SUBSCRIPTION_THREADS, new ThreadFactory() {
        @Override
        public Thread newThread(@NotNull Runnable r) {
            val t = Executors.defaultThreadFactory().newThread(r);
            t.setDaemon(true);
            return t;
        }
    });


    protected void createSubscription() {
        // create subscription
        ownSubscription = aeron.addSubscription("aeron:udp?endpoint=" + id() + ":" + voidConfiguration.getUnicastPort(), voidConfiguration.getStreamId());

        // create thread that polls messages from subscription
        messageHandler = new FragmentAssembler((buffer, offset, length, header) -> jointMessageHandler(buffer, offset, length, header));

        // starting thread(s) that will be fetching messages from network
        for (int e = 0; e < SUBSCRIPTION_THREADS; e++) {
            messagesExecutorService.execute(new Runnable() {
                @Override
                public void run() {
                    val idler = new SleepingIdleStrategy(1000);
                    while (true) {
                        idler.idle(ownSubscription.poll(messageHandler, 1024));
                    }
                }
            });
        }

        // starting thread(s) that will be actually executing message
        for (int e = 0; e < MESSAGE_THREADS; e++) {
            messagesExecutorService.execute(new Runnable() {
                @Override
                public void run() {
                    while (true) {
                        try {
                            // basically fetching messages from queue one by one, and processing them
                            val msg = messageQueue.take();
                            processMessage(msg);
                        } catch (InterruptedException e) {
                            // just terminate loop
                            break;
                        }
                    }
                }
            });
        }
    }

    /**
     * This method converts aeron buffer into VoidMessage and puts into temp queue for further processing
     *
     * @param buffer
     * @param offset
     * @param length
     * @param header
     */
    protected void jointMessageHandler(DirectBuffer buffer, int offset, int length, Header header) {
        byte[] data = new byte[length];
        buffer.getBytes(offset, data);

        // deserialize message
        val message = VoidMessage.fromBytes(data);

        // we're just putting deserialized message into the buffer
        try {
            messageQueue.put(message);
        } catch (InterruptedException e) {
            // :(
            throw new RuntimeException(e);
        }
    }

    protected void shutdownSilent() {
        // closing own connection
        ownSubscription.close();

        // and all known publications
        for (val rc: remoteConnections.values())
            rc.getPublication().close();

        // shutting down executor
        messagesExecutorService.shutdown();

        // closing aeron stuff
        aeron.close();
        context.close();
        driver.close();
    }

    @Override
    public void shutdown() {
        shutdownSilent();

        super.shutdown();
    }

    @Override
    public synchronized void launch() {
        // we set up aeron  connection to master first
        val id = mesh.get().getRootNode().getId();
        val port = voidConfiguration.getUnicastPort();

        // we add connection to the root node
        val v = aeron.addPublication("aeron:udp?endpoint=" + id + ":" + port, voidConfiguration.getStreamId());

        val hash = HashUtil.getLongHash(id + ":" + port);

        val rc = RemoteConnection.builder()
                .ip(id)
                .port(voidConfiguration.getUnicastPort())
                .longHash(hash)
                .publication(v)
                .build();

        remoteConnections.put(id, rc);

        // add own subscription
        createSubscription();

        super.launch();
    }

    @Override
    public synchronized void launchAsMaster() {
        // connection goes first, we're just creating subscription
        createSubscription();

        super.launchAsMaster();
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public void sendMessage(@NonNull VoidMessage message, @NonNull String id) {
        if (message.getOriginatorId() == null)
            message.setOriginatorId(this.id());

        // TODO: get rid of UUID!!!11
        if (message instanceof RequestMessage) {
            if (((RequestMessage) message).getRequestId() == null)
                ((RequestMessage) message).setRequestId(java.util.UUID.randomUUID().toString());
        }

        val conn = remoteConnections.get(id);
        if (conn == null)
            throw new ND4JIllegalStateException("Unknown target ID specified: [" + id + "]");

        // serialize & send message right away
        conn.getPublication().offer(message.asUnsafeBuffer());
    }

    @Data
    @Builder
    public static class RemoteConnection {
        private String ip;
        private int port;
        private Publication publication;
        private final Object locker = new Object();
        private final AtomicBoolean activated = new AtomicBoolean(false);
        protected long longHash;
    }

}
