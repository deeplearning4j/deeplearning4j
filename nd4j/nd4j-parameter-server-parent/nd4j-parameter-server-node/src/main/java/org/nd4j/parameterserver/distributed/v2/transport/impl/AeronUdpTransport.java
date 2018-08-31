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

import com.google.common.math.IntMath;
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
import org.agrona.CloseHelper;
import org.agrona.DirectBuffer;
import org.agrona.concurrent.SleepingIdleStrategy;
import org.jetbrains.annotations.NotNull;
import org.nd4j.base.Preconditions;
import org.nd4j.config.ND4JSystemProperties;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.HashUtil;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.enums.PropagationMode;
import org.nd4j.parameterserver.distributed.v2.enums.TransmissionStatus;
import org.nd4j.parameterserver.distributed.v2.messages.INDArrayMessage;
import org.nd4j.parameterserver.distributed.v2.messages.RequestMessage;
import org.nd4j.parameterserver.distributed.v2.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.handshake.HandshakeRequest;
import org.nd4j.parameterserver.distributed.v2.transport.MessageCallable;
import org.nd4j.parameterserver.distributed.v2.util.MeshOrganizer;
import org.nd4j.parameterserver.distributed.v2.util.MessageSplitter;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;

import static java.lang.System.setProperty;

/**
 * This class is a UDP implementation of Transport interface, based on Aeron
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class AeronUdpTransport extends BaseTransport implements AutoCloseable {
    // this is for tests only
    protected Map<String, MessageCallable> interceptors = new HashMap<>();
    protected Map<String, MessageCallable> precursors = new HashMap<>();

    // this map holds outgoing connections, basically
    protected Map<String, RemoteConnection> remoteConnections = new ConcurrentHashMap<>();

    protected final int SENDER_THREADS = 2;
    protected final int MESSAGE_THREADS = 2;
    protected final int SUBSCRIPTION_THREADS = 1;

    protected Aeron aeron;
    protected Aeron.Context context;

    protected Subscription ownSubscription;
    protected FragmentAssembler messageHandler;
    protected Thread subscriptionThread;

    // TODO: move this to singleton holder
    protected MediaDriver driver;

    private static final long DEFAULT_TERM_BUFFER_PROP = IntMath.pow(2,25); //32MB

    // this is intermediate buffer for incoming messages
    protected BlockingQueue<VoidMessage> messageQueue = new LinkedTransferQueue<>();

    // this is intermediate buffer for messages enqueued for propagation
    protected BlockingQueue<INDArrayMessage> propagationQueue = new LinkedBlockingQueue<>(32);

    // this lock is used for aeron publications
    protected ReentrantLock aeronLock = new ReentrantLock();

    protected final AtomicBoolean shutdownFlag = new AtomicBoolean(false);

    protected final AtomicBoolean connectedFlag = new AtomicBoolean(false);

    public AeronUdpTransport(@NonNull String ownIp, @NonNull String rootIp, @NonNull VoidConfiguration configuration) {
        this(ownIp, configuration.getUnicastPort(), rootIp, configuration.getUnicastPort(), configuration);
    }

    /**
     * This constructor creates root transport instance
     * @param rootIp
     * @param rootPort
     * @param configuration
     */
    public AeronUdpTransport(@NonNull String rootIp, int rootPort, @NonNull VoidConfiguration configuration) {
        this(rootIp, rootPort, rootIp, rootPort, configuration);
    }


    public AeronUdpTransport(@NonNull String ownIp, int ownPort, @NonNull String rootIp, int rootPort, @NonNull VoidConfiguration configuration) {
        super("aeron:udp?endpoint=" + ownIp + ":" + ownPort, "aeron:udp?endpoint=" + rootIp + ":" + rootPort, configuration);

        Preconditions.checkArgument(ownPort > 0 && ownPort < 65536, "Own UDP port should be positive value in range of 1 and 65536");
        Preconditions.checkArgument(rootPort > 0 && rootPort < 65536, "Master node UDP port should be positive value in range of 1 and 65536");

        setProperty("aeron.client.liveness.timeout", "30000000000");

        // setting this property to try to increase maxmessage length, not sure if it still works though
        //Term buffer length: must be power of 2 and in range 64kB to 1GB: https://github.com/real-logic/aeron/wiki/Configuration-Options
        String p = System.getProperty(ND4JSystemProperties.AERON_TERM_BUFFER_PROP);
        if(p == null){
            System.setProperty(ND4JSystemProperties.AERON_TERM_BUFFER_PROP, String.valueOf(DEFAULT_TERM_BUFFER_PROP));
        }

        splitter = MessageSplitter.getInstance();

        context = new Aeron.Context().driverTimeoutMs(30000)
                .keepAliveInterval(100000000);

        driver = MediaDriver.launchEmbedded();
        context.aeronDirectoryName(driver.aeronDirectoryName());
        aeron = Aeron.connect(context);

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            this.shutdown();
        }));
    }

    // this executor service han
    protected ExecutorService messagesExecutorService = Executors.newFixedThreadPool(SENDER_THREADS + MESSAGE_THREADS + SUBSCRIPTION_THREADS, new ThreadFactory() {
        @Override
        public Thread newThread(@NotNull Runnable r) {
            val t = Executors.defaultThreadFactory().newThread(r);
            t.setDaemon(true);
            return t;
        }
    });


    protected void createSubscription() {
        // create subscription
        ownSubscription = aeron.addSubscription(this.id() , voidConfiguration.getStreamId());

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


        for (int e = 0; e < SENDER_THREADS; e++) {
            messagesExecutorService.execute(new Runnable() {
                @Override
                public void run() {
                    while (true) {
                        try {
                            val msg = propagationQueue.take();
                            redirectedPropagateArrayMessage(msg);
                        } catch (InterruptedException e) {
                            break;
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
            });
        }
    }

    @Override
    protected void propagateArrayMessage(INDArrayMessage message, PropagationMode mode) throws IOException {
        try {
            propagationQueue.put(message);
        } catch (InterruptedException e) {
            // just swallow this
        }
    }

    protected void redirectedPropagateArrayMessage(INDArrayMessage message) throws IOException {
        super.propagateArrayMessage(message, PropagationMode.BOTH_WAYS);
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

        // we're checking if this is known connection or not, and add it if not
        if (!remoteConnections.containsKey(message.getOriginatorId()))
            addConnection(message.getOriginatorId());

        // we're just putting deserialized message into the buffer
        try {
            messageQueue.put(message);
        } catch (InterruptedException e) {
            // :(
            throw new RuntimeException(e);
        }
    }

    protected void addConnection(@NonNull String ipAndPort) {
        try {
            aeronLock.lock();

            if (remoteConnections.containsKey(ipAndPort))
                return;

            log.info("Adding UDP connection: [{}]", ipAndPort);

            val v = aeron.addPublication(ipAndPort, voidConfiguration.getStreamId());

            val hash = HashUtil.getLongHash(ipAndPort);

            val rc = RemoteConnection.builder()
                    .ip(ipAndPort)
                    .port(voidConfiguration.getUnicastPort())
                    .longHash(hash)
                    .publication(v)
                    .build();

            remoteConnections.put(ipAndPort, rc);
        } finally {
            aeronLock.unlock();
        }
    }

    @Override
    public void close() throws Exception {
        shutdown();
    }

    @Override
    public synchronized void launch() {
        if (!masterMode) {
            // we set up aeron  connection to master first
            addConnection(this.rootId);

            // add own subscription
            createSubscription();
        }

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
    public boolean isConnected() {
        if (connectedFlag.get() || masterMode)
            return true;

        // node supposed to be connected if rootNode is connected and downstreams + upstream + downstreams are connected
        if (!remoteConnections.containsKey(rootId))
            return false;

        synchronized (mesh) {
            val u = mesh.get().getUpstreamForNode(this.id()).getId();
            if (!remoteConnections.containsKey(u))
                return false;

            for (val n:mesh.get().getDownstreamsForNode(this.id())) {
                if (!remoteConnections.containsKey(n.getId()))
                    return false;
            }
        }

        connectedFlag.set(true);
        return true;
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

        // let's not send messages to ourselves
        if (message.getOriginatorId().equals(id)) {
            this.processMessage(message);
            return;
        }

        // serialize out of locks
        val b = message.asUnsafeBuffer();

        // blocking until all connections are up
        if (!id.equals(rootId)) {
            while (!isConnected()) {
                LockSupport.parkNanos(10000000);
            }
        }

        val conn = remoteConnections.get(id);
        if (conn == null)
            throw new ND4JIllegalStateException("Unknown target ID specified: [" + id + "]");

        // serialize & send message right away
        TransmissionStatus status = TransmissionStatus.UNKNOWN;
        while (status != TransmissionStatus.OK) {
            synchronized (conn.locker) {
                status = TransmissionStatus.fromLong(conn.getPublication().offer(b));
            }

            // if response != OK we must do something with response
            switch (status) {
                case MAX_POSITION_EXCEEDED:
                case CLOSED: {
                    // TODO: here we should properly handle reconnection
                    log.warn("Upstream connection was closed: [{}]", id);
                    return;
                }
                case ADMIN_ACTION:
                case NOT_CONNECTED:
                case BACK_PRESSURED: {
                    try {
                        // in case of backpressure we're just sleeping for a while, and message out again
                        Thread.sleep(voidConfiguration.getRetransmitTimeout());
                    } catch (InterruptedException e) {
                        //
                    }
                }
            }
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
        if (shutdownFlag.compareAndSet(false, true)) {
            shutdownSilent();

            super.shutdown();
        }
    }

    @Override
    public  void onMeshUpdate(MeshOrganizer mesh) {
        mesh.flatNodes().forEach(n -> addConnection(n.getId()));

        super.onMeshUpdate(mesh);
    }

    /**
     * This method add interceptor for incoming messages. If interceptor is defined for given message class - runnable will be executed instead of processMessage()
     * @param cls
     * @param callable
     */
    public <T extends VoidMessage> void addInterceptor(@NonNull Class<T> cls, @NonNull MessageCallable<T> callable) {
        interceptors.put(cls.getCanonicalName(), callable);
    }

    /**
     * This method add precursor for incoming messages. If precursor is defined for given message class - runnable will be executed before processMessage()
     * @param cls
     * @param callable
     */
    public <T extends VoidMessage> void addPrecursor(@NonNull Class<T> cls, @NonNull MessageCallable<T> callable) {
        precursors.put(cls.getCanonicalName(), callable);
    }

    @Override
    public void processMessage(@NonNull VoidMessage message) {
        // fast super call if there's no callbacks where defined
        if (interceptors.isEmpty() && precursors.isEmpty()) {
            super.processMessage(message);
            return;
        }

        val name = message.getClass().getCanonicalName();
        val callable = interceptors.get(name);

        if (callable != null)
            callable.apply(message);
        else {
            val precursor = precursors.get(name);
            if (precursor != null)
                precursor.apply(message);

            super.processMessage(message);
        }
    }

    /**
     * This method returns Mesh stored in this Transport instance
     * PLEASE NOTE: This method is suited for tests
     * @return
     */
    protected MeshOrganizer getMesh() {
        synchronized (mesh) {
            return mesh.get();
        }
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
