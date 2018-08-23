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
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.primitives.Atomic;
import org.nd4j.linalg.primitives.Optional;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.enums.MeshBuildMode;
import org.nd4j.parameterserver.distributed.v2.chunks.VoidChunk;
import org.nd4j.parameterserver.distributed.v2.enums.PropagationMode;
import org.nd4j.parameterserver.distributed.v2.messages.*;
import org.nd4j.parameterserver.distributed.v2.messages.impl.MeshUpdateMessage;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.handshake.HandshakeRequest;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.handshake.HandshakeResponse;
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
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

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

    protected BaseTransport() {
        this(java.util.UUID.randomUUID().toString());
    }

    protected BaseTransport(@NonNull String rootId) {
        this(rootId, VoidConfiguration.builder().build());
    }

    protected BaseTransport(@NonNull String rootId, @NonNull VoidConfiguration voidConfiguration) {
        this.mesh.set(new MeshOrganizer(meshBuildMode));
        this.rootId = rootId;
        this.voidConfiguration = voidConfiguration;
    }

    protected BaseTransport(@NonNull String ownId, @NonNull String rootId, @NonNull VoidConfiguration voidConfiguration) {
        this.mesh.set(new MeshOrganizer(meshBuildMode));
        this.id = ownId;
        this.rootId = rootId;
        this.voidConfiguration = voidConfiguration;

        masterMode = ownId.equalsIgnoreCase(rootId);
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
        // first of all we introduce ourselves to master

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

        // now we're going for Handshake
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
        this.launch();
    }

    @Override
    public synchronized void shutdown() {
        // probably will be nothing useful in this implementation
    }

    @Override
    public void propagateMessage(@NonNull VoidMessage voidMessage, PropagationMode mode) throws IOException {
        val node = mesh.get().getNodeById(id);

        //if (voidMessage.getOriginatorId() != null && id != null && voidMessage.getOriginatorId().equals(id))
         //   return;

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
            // TODO: make chunk size configurable
            val chunks = MessageSplitter.getInstance().split(voidMessage, 65536);
            // send chunks to the upstream
            if (!node.isRootNode() && (PropagationMode.BOTH_WAYS == mode || PropagationMode.ONLY_UP == mode))
                chunks.forEach(c -> sendMessage(c, upstream.getId()));

            // and send chunks to all downstreams
            if (PropagationMode.BOTH_WAYS == mode || PropagationMode.ONLY_DOWN == mode)
                downstreams.parallelStream().forEach(n -> {
                    chunks.forEach(c -> sendMessage(c, n.getId()));
                });
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
        val node = mesh.get().getNodeById(id);

        if (voidMessage.getOriginatorId() != null && id != null && voidMessage.getOriginatorId().equals(id))
            return;

        val root = mesh.get().getRootNode();
        val upstream = node.getUpstreamNode();
        val downstreams = node.getDownstreamNodes();

        // we never propagate upstream if we're on root node
        // we never send to the latest node
        // we never send to the original node
        if (!node.isRootNode() && (PropagationMode.BOTH_WAYS == mode || PropagationMode.ONLY_UP == mode) && !isLoopedNode(upstream, voidMessage)) {
            voidMessage.setRelayId(id);
            sendMessage(voidMessage, upstream.getId());
        }

        // now we're sending message down
        if (PropagationMode.BOTH_WAYS == mode || PropagationMode.ONLY_DOWN == mode) {
            voidMessage.setRelayId(id);
            downstreams.forEach(n -> {
                if (!isLoopedNode(n, voidMessage))
                    sendMessage(voidMessage, n.getId());
            });
        }
    }

    protected boolean isLoopedNode(@NonNull MeshOrganizer.Node node, @NonNull BroadcastableMessage message) {
        return node.getId().equals(message.getOriginatorId())
                || node.getId().equals(message.getRelayId());
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

    @Override
    public void processMessage(VoidMessage message) {
        /**
         * TODO: we need better isolation here
         */
        if (message instanceof VoidChunk) {
            // we merge chunks to get full INDArrayMessage
            Optional<INDArrayMessage> opt = MessageSplitter.getInstance().merge((VoidChunk) message);

            // if this chunk was the last message, we'll forward it to parameter server for actual use
            if (opt.isPresent())
                this.processMessage(opt.get());
        } else if (message instanceof INDArrayMessage) {
            // just forward message, but ONLY if it's not a Response message, since it's probably processed separately
            if (!(message instanceof ResponseMessage))
                forwardToParameterServer((INDArrayMessage) message);
            else {
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
                    // first we add new node to the mesh
                    val nodeToRemap = mesh.get().getNodeById(message.getOriginatorId());
                    val nodesToRemap = new ArrayList<MeshOrganizer.Node>();

                    // we're remapping node, and its downstreams
                    nodesToRemap.add(nodeToRemap);
                    nodesToRemap.addAll(nodeToRemap.getDownstreamNodes());

                    for (val n : nodesToRemap)
                        mesh.get().remapNode(n);

                    // we don't want remapped node to have any downstreams
                    nodeToRemap.truncateDownstreams();

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
                propagateMessage(new MeshUpdateMessage(mesh.get()), PropagationMode.ONLY_DOWN);
            } catch (Exception e) {
                log.error("Wasn't able to propagate message from [{}]", id());
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
                    onMeshUpdate(newMesh);
                }
            }
        } else {
            if (message instanceof RequestMessage) {
                val name = message.getClass().getCanonicalName();
                val consumer = consumers.get(name);
                if (consumer == null)
                    throw new ND4JIllegalStateException("Unknown message received: [" + message.getClass().getCanonicalName() + "]");
            } else
                throw new ND4JIllegalStateException("Unknown message received: [" + message.getClass().getCanonicalName() + "]");
        }


        if (message instanceof BroadcastableMessage) {
            // here we should propagate message down
            try {
                propagateBroadcastableMessage((BroadcastableMessage) message, PropagationMode.ONLY_DOWN);
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
        // no-op
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
}
