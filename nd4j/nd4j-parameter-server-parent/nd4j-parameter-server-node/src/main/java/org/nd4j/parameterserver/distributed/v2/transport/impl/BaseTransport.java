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

import io.reactivex.Flowable;
import io.reactivex.functions.Consumer;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.primitives.Atomic;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.linalg.primitives.Optional;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.enums.MeshBuildMode;
import org.nd4j.parameterserver.distributed.v2.chunks.VoidChunk;
import org.nd4j.parameterserver.distributed.v2.enums.PropagationMode;
import org.nd4j.parameterserver.distributed.v2.messages.*;
import org.nd4j.parameterserver.distributed.v2.messages.history.HashHistoryHolder;
import org.nd4j.parameterserver.distributed.v2.messages.impl.MeshUpdateMessage;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.handshake.HandshakeRequest;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.handshake.HandshakeResponse;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.ping.PingMessage;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.ping.PongMessage;
import org.nd4j.parameterserver.distributed.v2.messages.MessagesHistoryHolder;
import org.nd4j.parameterserver.distributed.v2.transport.RestartCallback;
import org.nd4j.parameterserver.distributed.v2.transport.Transport;
import org.nd4j.parameterserver.distributed.v2.util.MeshOrganizer;
import org.nd4j.parameterserver.distributed.v2.util.MessageSplitter;
import org.reactivestreams.Publisher;
import org.reactivestreams.Subscriber;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

/**
 *
 * @author raver119@gmail.com
 */
@Slf4j
public abstract  class BaseTransport  implements Transport {
    // this stream is for delivering messages from this host to other hosts in the network
    protected final MessageFlow<VoidMessage> outgoingFlow = new MessageFlow<>();

    // this stream is for receiving INDArray messages from the network
    protected final MessageFlow<INDArrayMessage> incomingFlow = new MessageFlow<>();

    // here we're storing reference to mesh
    protected final Atomic<MeshOrganizer> mesh = new Atomic<>();

    // this is Id of this Transport instance
    protected String id;

    // id of the root node is used for initial communication
    protected String rootId;

    protected boolean masterMode = false;

    // this is simple storage for replies
    protected final Map<String, ResponseMessage> replies = new ConcurrentHashMap<>();

    // dedicated callback for restart messages
    protected RestartCallback restartCallback;

    // collection of callbacks for connection with ParameterServer implementation
    protected Map<String, Consumer> consumers = new HashMap<>();

    // just configuration bean
    protected final VoidConfiguration voidConfiguration;

    protected final MeshBuildMode meshBuildMode = MeshBuildMode.MESH;

    // exactly what name says
    protected final AtomicInteger numerOfNodes = new AtomicInteger(0);

    // this queue handles all incoming messages
    protected final TransferQueue<VoidMessage> messageQueue = new LinkedTransferQueue<>();

    // MessageSplitter instance that'll be used in this transport
    protected MessageSplitter splitter;

    // we're keeping Ids of last 2k INDArrayMessages, just to avoid double spending/retransmission
    protected MessagesHistoryHolder<String> historyHolder = new HashHistoryHolder<String>(2048);

    protected final ThreadPoolExecutor executorService = (ThreadPoolExecutor) Executors.newFixedThreadPool(Math.max(2, Runtime.getRuntime().availableProcessors()), new ThreadFactory() {
        @Override
        public Thread newThread(@NotNull Runnable r) {
            val t = Executors.defaultThreadFactory().newThread(r);
            t.setDaemon(true);
            return t;
        }
    });



    protected BaseTransport() {
        this(java.util.UUID.randomUUID().toString());
    }

    protected BaseTransport(@NonNull String rootId) {
        this(rootId, VoidConfiguration.builder().build());
    }

    protected BaseTransport(@NonNull String rootId, @NonNull VoidConfiguration voidConfiguration) {
        this.mesh.set(new MeshOrganizer(voidConfiguration.getMeshBuildMode()));
        this.rootId = rootId;
        this.voidConfiguration = voidConfiguration;
    }

    protected BaseTransport(@NonNull String ownId, @NonNull String rootId, @NonNull VoidConfiguration voidConfiguration) {
        this.mesh.set(new MeshOrganizer(voidConfiguration.getMeshBuildMode()));
        this.id = ownId;
        this.rootId = rootId;
        this.voidConfiguration = voidConfiguration;

        masterMode = ownId.equalsIgnoreCase(rootId);
        if (masterMode) {
            this.mesh.get().getRootNode().setId(rootId);
        }
    }

    @Override
    public Consumer<VoidMessage> outgoingConsumer() {
        return outgoingFlow;
    }

    @Override
    public Publisher<INDArrayMessage> incomingPublisher() {
        return incomingFlow;
    }

    @Override
    public String getUpstreamId() {
        if (mesh.get().getRootNode().getId().equals(this.id()))
            return this.id();

        return mesh.get().getNodeById(this.id()).getUpstreamNode().getId();
    }

    @Override
    public synchronized void launch() {
        // master mode assumes heartbeat thread, so we'll need one more thread to run there
        int lim = masterMode ? 1 : 0;
        // we're launching threads for messages processing
        for (int e = 0; e< executorService.getMaximumPoolSize() - lim; e++) {
            executorService.submit(new Runnable() {
                @Override
                public void run() {
                    while (true) {
                        try {
                            val message = messageQueue.take();
                            if (message != null)
                                internalProcessMessage(message);
                        } catch (InterruptedException e) {
                            break;
                        } catch (Exception e) {
                            log.error("Exception: {}", e);
                        }
                    }
                }
            });
        }


        // this flow gets converted to VoidChunks and sent to upstream and downstreams
        val d = Flowable.fromPublisher(outgoingFlow).subscribe(voidMessage -> {
            if (mesh.get() == null) {
                log.warn("Mesh wasn't received yet!");
                return;
            }

            // we're tagging this message as originated locally
            voidMessage.setOriginatorId(id);

            // and propagating message across mesh network
            propagateMessage(voidMessage, PropagationMode.BOTH_WAYS);
        });

        // now we're going for Handshake with master
        if (!masterMode) {
            try {
                sendMessageBlocking(new HandshakeRequest(), rootId);
            } catch (Exception e) {
                throw new ND4JIllegalStateException("Can't proceed with handshake from [" + this.id() + "] to [" + rootId + "]", e);
            }
        }
    }

    @Override
    public synchronized void launchAsMaster() {
        if (mesh.get() == null)
            mesh.set(new MeshOrganizer(meshBuildMode));

        masterMode = true;
        mesh.get().getRootNode().setId(this.id());

        // launching heartbeat thread, that will monitor offline nodes
        executorService.submit(new HeartbeatThread(120000, this, mesh));

        this.launch();
    }

    @Override
    public synchronized void shutdown() {
        // shuttng down
        executorService.shutdown();
    }

    protected void propagateArrayMessage(INDArrayMessage message, PropagationMode mode) throws IOException  {
        val node = mesh.get().getNodeById(id);

        val root = mesh.get().getRootNode();
        val upstream = node.getUpstreamNode();
        val downstreams = node.getDownstreamNodes();

        // TODO: make chunk size configurable
        val chunks = splitter.split(message, voidConfiguration.getMaxChunkSize());
        // send chunks to the upstream
        if (!node.isRootNode() && (PropagationMode.BOTH_WAYS == mode || PropagationMode.ONLY_UP == mode))
            chunks.forEach(c -> sendMessage(c, upstream.getId()));

        // and send chunks to all downstreams
        if (PropagationMode.BOTH_WAYS == mode || PropagationMode.ONLY_DOWN == mode)
            downstreams.parallelStream().forEach(n -> {
                chunks.forEach(c -> sendMessage(c, n.getId()));
            });
    }

    @Override
    public void propagateMessage(@NonNull VoidMessage voidMessage, PropagationMode mode) throws IOException {
        val node = mesh.get().getNodeById(id);

        //if (voidMessage.getOriginatorId() != null && id != null && voidMessage.getOriginatorId().equals(id))
         //   return;

        // it's possible situation to have master as regular node. i.e. spark localhost mode
        if (mesh.get().totalNodes() == 1) {
            internalProcessMessage(voidMessage);
            return;
        }

        val root = mesh.get().getRootNode();
        val upstream = node.getUpstreamNode();
        val downstreams = node.getDownstreamNodes();

        // setting on first one
        //if (voidMessage.getOriginatorId() == null)
            //voidMessage.setOriginatorId(this.id());

        if (voidMessage instanceof BroadcastableMessage) {
            ((BroadcastableMessage) voidMessage).setRelayId(id);
        }

        // if this is INDArrayMessage we'll split it into chunks
        if (voidMessage instanceof INDArrayMessage) {
            propagateArrayMessage((INDArrayMessage) voidMessage, mode);
        } else {
            // send message to the upstream
            if (!node.isRootNode() && (PropagationMode.BOTH_WAYS == mode || PropagationMode.ONLY_UP == mode))
                sendMessage(voidMessage, upstream.getId());

            // and send message for all downstreams
            if (PropagationMode.BOTH_WAYS == mode || PropagationMode.ONLY_DOWN == mode)
                downstreams.forEach(n -> sendMessage(voidMessage, n.getId()));
        }
    }

    protected void propagateBroadcastableMessage(@NonNull BroadcastableMessage voidMessage, PropagationMode mode) {
        // we never broadcast MeshUpdates, master will send everyone copy anyway
        if (voidMessage instanceof MeshUpdateMessage)
           return;

        // if this message is already a known one - just skip it
        if (historyHolder.storeIfUnknownMessageId(voidMessage.getMessageId()))
            return;

        val node = mesh.get().getNodeById(id);

        if (voidMessage.getOriginatorId() != null && id != null && voidMessage.getOriginatorId().equals(id))
            return;

        val root = mesh.get().getRootNode();
        val upstream = node.getUpstreamNode();
        val downstreams = node.getDownstreamNodes();

        val ownId = id();
        val upstreamId = node.isRootNode() ? null : upstream.getId();
        val originatorId = voidMessage.getOriginatorId();
        val relayId = voidMessage.getRelayId();
        voidMessage.setRelayId(id());

        // we never propagate upstream if we're on root node
        // we never send to the latest node
        // we never send to the original node
        if (!node.isRootNode() && (PropagationMode.BOTH_WAYS == mode || PropagationMode.ONLY_UP == mode) && !isLoopedNode(upstream, originatorId, relayId)) {
            if (!isLoopedNode(upstream, originatorId, relayId)) {
                sendMessage(voidMessage, upstreamId);
            }
        }

        // now we're sending message down
        if (PropagationMode.BOTH_WAYS == mode || PropagationMode.ONLY_DOWN == mode) {
            downstreams.forEach(n -> {
                if (!isLoopedNode(n, originatorId, relayId)) {
                    sendMessage(voidMessage, n.getId());
                }
            });
        }
    }

    protected boolean isLoopedNode(@NonNull MeshOrganizer.Node node, @NonNull String originatorId, @NonNull String relayId) {
        return node.getId().equals(originatorId) || node.getId().equals(relayId);
    }

    /**
     * This method puts INDArray to the flow read by parameter server
     * @param message
     */
    private void forwardToParameterServer(INDArrayMessage message) {
        try {
            incomingFlow.accept(message);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    protected void internalProcessMessage(VoidMessage message) {
        /**
         * TODO: we need better isolation here
         */
        if (message instanceof PingMessage) {

            val msg = new PongMessage();
            msg.setRequestId(((PingMessage) message).getRequestId());
            sendMessage(msg, message.getOriginatorId());
        } if (message instanceof PongMessage) {

            // do nothing
        }  else if (message instanceof VoidChunk) {
            // we merge chunks to get full INDArrayMessage
            Optional<INDArrayMessage> opt = splitter.merge((VoidChunk) message, voidConfiguration.getChunksBufferSize());

            // if this chunk was the last message, we'll forward it to parameter server for actual use
            if (opt.isPresent())
                this.processMessage(opt.get());
        } else if (message instanceof INDArrayMessage) {
            // just forward message, but ONLY if it's not a Response message, since it's probably processed separately
            if (!(message instanceof ResponseMessage)) {
                if (!historyHolder.isKnownMessageId(message.getMessageId())) {// we're not applying the same message twice
                    forwardToParameterServer((INDArrayMessage) message);
                }
            } else {
                // in this case we store message to the map, to be fetched later
                val reply = (ResponseMessage) message;
                replies.putIfAbsent(reply.getRequestId(), reply);
            }
        } else if (message instanceof HandshakeRequest) {
            synchronized (mesh) {
                if (!mesh.get().isKnownNode(this.id())) {
                    mesh.get().getRootNode().setId(this.id);
                }
            }

            // our response
            val response = HandshakeResponse.builder()
                    .build();

            synchronized (mesh) {
                if (mesh.get().isKnownNode(message.getOriginatorId())) {
                    mesh.get().remapNodeAndDownstreams(message.getOriginatorId());
                    // we say that this model has restarted
                    response.setRestart(true);
                } else {
                    // first we add new node to the mesh
                    mesh.get().addNode(message.getOriginatorId());
                }

                response.setMesh(mesh.get().clone());
            }

            response.setRequestId(((HandshakeRequest) message).getRequestId());
            sendMessage(response, message.getOriginatorId());

            // update all other nodes with new mesh
            try {
                propagateMessageDirect(new MeshUpdateMessage(mesh.get()));
            } catch (Exception e) {
                log.error("Wasn't able to propagate message from [{}]", id());
                log.error("Exception: {}", e);
                throw new RuntimeException(e);
            }
        } else if (message instanceof HandshakeResponse) {
            val response = (HandshakeResponse) message;
            val newMesh = response.getMesh();

            mesh.cas(null, response.getMesh());

            synchronized (mesh) {
                val v1 = mesh.get().getVersion();
                val v2 = newMesh.getVersion();

                //log.info("Starting update A on [{}]; version: [{}/{}]; size: [{}]", this.id(), v1, v2, newMesh.totalNodes());
                // we update only if new mesh is older that existing one
                if (v1 < v2)
                    mesh.set(newMesh);
            }

            // optionally calling out callback, which will happen approximately 100% of time
            if (response.isRestart()) {
                if (restartCallback != null)
                    restartCallback.call(response);
                else
                    log.warn("Got restart message from master, but there's no defined RestartCallback");
            }


            // in any way we're putting this message back to replies
            val reply = (ResponseMessage) message;
            replies.putIfAbsent(reply.getRequestId(), reply);

            // this is default handler for message pairs
        } else if (message instanceof ResponseMessage) {
            // in this case we store message to the map, to be fetched later
            val reply = (ResponseMessage) message;
            replies.putIfAbsent(reply.getRequestId(), reply);

        } else if (message instanceof MeshUpdateMessage) {
            val newMesh = ((MeshUpdateMessage) message).getMesh();

            mesh.cas(null, newMesh);

            synchronized (mesh) {
                val v1 = mesh.get().getVersion();
                val v2 = newMesh.getVersion();

                //log.info("Starting update B on [{}]; version: [{}/{}]; size: [{}]", this.id(), v1, v2, newMesh.totalNodes());
                // we update only if new mesh is older that existing one
                if (v1 < v2) {
                    mesh.set(newMesh);
                }
            }

            // should be out of locked block
            onMeshUpdate(newMesh);
        } else {
            if (message instanceof RequestMessage) {
                val name = message.getClass().getCanonicalName();
                val consumer = consumers.get(name);
                if (consumer == null)
                    throw new ND4JIllegalStateException("Not supported  RequestMessage received: [" + message.getClass().getCanonicalName() + "]");
            } else
                throw new ND4JIllegalStateException("Unknown message received: [" + message.getClass().getCanonicalName() + "]");
        }


        if (message instanceof BroadcastableMessage) {
            // here we should propagate message down
            try {
                propagateBroadcastableMessage((BroadcastableMessage) message, PropagationMode.BOTH_WAYS);
            } catch (Exception e) {
                log.error("Wasn't able to propagate message from [{}]", id());
                throw new RuntimeException(e);
            }
        }

        // Request messages might be sent back to ParameterServer, which will take care of processing
        if (message instanceof RequestMessage) {
            // looks for callback for a given message type
            val consumer = consumers.get(message.getClass().getCanonicalName());
            if (consumer != null) {
                try {
                    consumer.accept(message);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    public void propagateMessageDirect(@NonNull BroadcastableMessage message) {
        synchronized (mesh) {
            val nodes = mesh.get().flatNodes();
            nodes.stream().forEach(n -> {
                if (!n.isRootNode())
                    sendMessage(message, n.getId());
            });
        }
    }

    @Override
    public void processMessage(VoidMessage message) {
        try {
            messageQueue.transfer(message);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public <T extends ResponseMessage> T sendMessageBlocking(@NonNull RequestMessage message, @NonNull String id) throws InterruptedException {
        if (message.getRequestId() == null)
            message.setRequestId(java.util.UUID.randomUUID().toString());

        // we send message to the node first
        sendMessage(message, id);

        // and then we just block until we get response
        ResponseMessage r = null;
        while ((r = replies.get(message.getRequestId())) == null) {
            Thread.sleep(10);
        }

        // remove response from holder
        replies.remove(message.getRequestId());

        //and return reply back
        return (T) r;
    }


    @Override
    public <T extends ResponseMessage> T sendMessageBlocking(@NonNull RequestMessage message, @NonNull String id, long timeWait, @NonNull TimeUnit timeUnit) throws InterruptedException {
        if (message.getRequestId() == null)
            message.setRequestId(java.util.UUID.randomUUID().toString());

        // we send message to the node first
        sendMessage(message, id);

        val sleepMs = TimeUnit.MILLISECONDS.convert(timeWait, timeUnit);
        val startTime = System.currentTimeMillis();

        // and then we just block until we get response
        ResponseMessage r = null;
        while ((r = replies.get(message.getRequestId())) == null) {
            val currTime = System.currentTimeMillis();

            if (currTime - startTime > sleepMs)
                break;

            LockSupport.parkNanos(5000);
        }



        // remove response from holder
        replies.remove(message.getRequestId());

        //and return reply back
        return (T) r;
    }

    @Override
    public void setRestartCallback(RestartCallback callback) {
        this.restartCallback = callback;
    }

    @Override
    public <T extends RequestMessage> void addRequestConsumer(@NonNull Class<T> cls, Consumer<T> consumer) {
        if (consumer == null)
            consumers.remove(cls.getCanonicalName());
        else
            consumers.put(cls.getCanonicalName(), consumer);
    }

    @Override
    public void onMeshUpdate(MeshOrganizer mesh) {
        // FIXME: (int) is bad here
        numerOfNodes.set((int) mesh.totalNodes());
    }

    /**
     * Generic Publisher/Consumer implementation for interconnect
     * @param <T>
     */
    public static class MessageFlow<T> implements Consumer<T>, Publisher<T> {
        private List<Subscriber<? super T>> subscribers = new CopyOnWriteArrayList<>();

        @Override
        public void accept(T voidMessage) throws Exception {
            // just propagate messages further away
            subscribers.forEach(s -> s.onNext(voidMessage));
        }

        @Override
        public void subscribe(Subscriber<? super T> subscriber) {
            // we're just maintaining list of
            subscribers.add(subscriber);
        }
    }


    protected static class HeartbeatThread extends Thread implements Runnable {
        protected final long delay;
        protected final Atomic<MeshOrganizer> mesh;
        protected final Transport transport;

        protected HeartbeatThread(long delayMilliseconds, @NonNull Transport transport, @NonNull Atomic<MeshOrganizer> mesh) {
            this.delay = delayMilliseconds;
            this.mesh = mesh;
            this.transport = transport;
        }

        @Override
        public void run() {
            try {
                while (true) {
                    Thread.sleep(delay);
                    val remapped = new AtomicBoolean(false);

                    val nodes = mesh.get().flatNodes();
                    for (val n : nodes) {
                        // we're skipping own node
                        if (transport.id().equals(n.getId()))
                            continue;

                        PongMessage m = transport.sendMessageBlocking(new PingMessage(), n.getId(), 100, TimeUnit.MILLISECONDS);

                        // if we're not getting response in reasonable time - we're considering this node as failed
                        if (m == null) {
                            mesh.get().remapNode(n);
                            mesh.get().markNodeOffline(n);
                            remapped.set(true);
                        }
                    }

                    if (remapped.get()) {
                        try {
                            transport.propagateMessage(new MeshUpdateMessage(mesh.get()), PropagationMode.ONLY_DOWN);
                        } catch (IOException e) {
                            // hm.
                        }
                    }
                }
            } catch (InterruptedException e) {
                //
            }
        }
    }

    @Override
    public String getRootId() {
        return rootId;
    }

    @Override
    public int totalNumberOfNodes() {
        return numerOfNodes.get();
    }
}
