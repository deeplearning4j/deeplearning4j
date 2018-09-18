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

package org.nd4j.parameterserver.distributed.v2;

import io.reactivex.Flowable;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Atomic;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.enums.PropagationMode;
import org.nd4j.parameterserver.distributed.v2.messages.impl.GradientsUpdateMessage;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.handshake.HandshakeResponse;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.params.ModelParametersMessage;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.params.ModelParametersRequest;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.params.UpdaterParametersMessage;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.params.UpdaterParametersRequest;
import org.nd4j.parameterserver.distributed.v2.transport.RestartCallback;
import org.nd4j.parameterserver.distributed.v2.transport.Transport;
import org.nd4j.parameterserver.distributed.v2.transport.UpdaterParametersProvider;
import org.nd4j.parameterserver.distributed.v2.transport.UpdatesHandler;
import org.nd4j.parameterserver.distributed.v2.util.AbstractSubscriber;
import org.nd4j.parameterserver.distributed.v2.util.MeshOrganizer;
import org.nd4j.parameterserver.distributed.v2.util.UpdaterParametersHolder;
import org.reactivestreams.Subscriber;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 *
 */
@Slf4j
public final class ModelParameterServer {
    protected static final ModelParameterServer INSTANCE = new ModelParameterServer();

    @Getter
    private Transport transport;

    @Getter
    private INDArray masterModelParams;

    @Getter
    private INDArray masterUpdaterParams;

    private UpdaterParametersProvider updaterParametersProvider;

    // queue is used only if there's no subscribers defined
    private final BlockingQueue<INDArray> updatesQueue = new LinkedBlockingQueue<>(4096);

    // subsribers that are connected to actual model
    protected final List<UpdatesHandler> updatesSubscribers = new CopyOnWriteArrayList<>();
    protected final List<Subscriber<INDArray>> modelParamsSubsribers = new CopyOnWriteArrayList<>();
    protected final List<Subscriber<INDArray>> updaterParamsSubscribers = new CopyOnWriteArrayList<>();

    private boolean masterMode;

    protected VoidConfiguration configuration;

    // this flag is true once mps is launched
    private final AtomicBoolean launchLock = new AtomicBoolean(false);
    private final AtomicBoolean stopLock = new AtomicBoolean(false);

    // this queue is used as temporary storage for updates received during restart event.
    protected BlockingQueue<INDArray> updatesBacklog = new LinkedBlockingQueue<>();

    // these two fields only used at master node, to store latest updater copy
    protected final Atomic<UpdaterParametersHolder> updaterParameters = new Atomic<>();
    protected final ReentrantReadWriteLock updaterParamsLock = new ReentrantReadWriteLock();
    protected final AtomicBoolean gotFinalState = new AtomicBoolean(false);

    private Disposable disposable;

    protected ModelParameterServer() {
        //
    }

    public static ModelParameterServer getInstance() {
        return INSTANCE;
    }

    /**
     * This constructor is for tests only
     *
     * @param transport
     */
    protected ModelParameterServer(@NonNull Transport transport) {
        this(transport, false);
    }

    /**
     * This constructor is for tests only
     *
     * @param transport
     * @param isMasterNode
     */
    protected ModelParameterServer(@NonNull Transport transport, boolean isMasterNode) {
        this(VoidConfiguration.builder().unicastPort(40123).streamId(119).build(), transport, isMasterNode);
    }

    /**
     * This constructor creates new ModelParameterServer instance
     *
     * @param configuration VoidConfiguration bean
     * @param transport Transport instance to be used for communications
     * @param isMasterNode set to true if this parameter server instance will be a master node, false otherwise
     */
    public ModelParameterServer(@NonNull VoidConfiguration configuration, @NonNull Transport transport, boolean isMasterNode) {
        this();
        configure(configuration, transport, isMasterNode);
    }

    /**
     * This method stores provided entities for MPS internal use
     *
     * @param configuration
     * @param transport
     * @param isMasterNode
     */
    public void configure(@NonNull VoidConfiguration configuration, @NonNull Transport transport, boolean isMasterNode) {
        this.transport = transport;
        this.masterMode = isMasterNode;
        this.configuration = configuration;
    }

    /**
     * This method stores provided entities for MPS internal use
     *
     * @param configuration
     * @param transport
     * @param isMasterNode
     */
    public void configure(@NonNull VoidConfiguration configuration, @NonNull Transport transport, @NonNull UpdaterParametersProvider updaterProvider) {
        this.transport = transport;
        this.masterMode = false;
        this.configuration = configuration;
        this.updaterParametersProvider = updaterProvider;
    }

    /**
     * This method adds subcriber that will be called upon gradients update receival
     * @param s
     */
    public void addUpdatesSubscriber(@NonNull UpdatesHandler s) {
        updatesSubscribers.add(s);
    }

    /**
     * This method adds subcriber that will be called upon model params receival
     * @param s
     */
    public void addModelParamsSubscriber(@NonNull Subscriber<INDArray> s) {
        modelParamsSubsribers.add(s);
    }

    /**
     * This method adds subcriber that will be called upon updater params receival
     * @param s
     */
    public void addUpdaterParamsSubscriber(@NonNull Subscriber<INDArray> s) {
        updaterParamsSubscribers.add(s);
    }

    /**
     * This method checks if ModelParameterServer was initialized
     *
     * @return true if already initalized, false otherwise
     */
    public boolean isInitialized() {
        return launchLock.get();
    }

    /**
     * This method starts parameter server
     */
    public synchronized void launch() {
        log.info("ModelParameterServer starts");
        if (launchLock.get())
            return;

        transport.setRestartCallback(new RestartCallback() {
            @Override
            public void call(HandshakeResponse response) {
                // upon restart command we'll request current parameters from the current upstream (without any propagation
                try {
                    log.info("Restart callback started...");
                    val msg = new ModelParametersRequest();
                    val rootId = transport.getRootId();
                    ModelParametersMessage modelParams = transport.sendMessageBlocking(msg, rootId);
                    val mParams = modelParams.getPayload();
                    modelParamsSubsribers.forEach(s -> s.onNext(mParams));

                    // updater parameters are optional, it's possible to have models without updater parameters (i.e. SGD)
                    UpdaterParametersMessage updaterParams = transport.sendMessageBlocking(new UpdaterParametersRequest(), rootId);
                    val uParams = updaterParams.getPayload();
                    if (uParams != null) {
                        updaterParamsSubscribers.forEach(s -> s.onNext(uParams));
                        log.info("Updater parameters propagated...");
                    }
                } catch (Exception e) {
                    log.error("RestartCallback processing exception: {}", e);
                    throw new RuntimeException(e);
                }
            }
        });

        // listener for model params requests
        transport.addRequestConsumer(ModelParametersRequest.class, new Consumer<ModelParametersRequest>() {
            @Override
            public void accept(ModelParametersRequest modelParametersRequest) throws Exception {
                // send model parameters somewhere
                val msg = new ModelParametersMessage(java.util.UUID.randomUUID().toString(), updatesSubscribers.get(0).getParametersArray());
                msg.setRequestId(modelParametersRequest.getRequestId());
                transport.sendMessage(msg, modelParametersRequest.getOriginatorId());
            }
        });

        if (masterMode) {
            // on master node when updater params come - we're just storing them
            addUpdaterParamsSubscriber(new AbstractSubscriber<INDArray>() {
                @Override
                public void onNext(INDArray array) {
                    // we're keeping first final updater params state
                    if (gotFinalState.get())
                        return;
                    try {
                        updaterParamsLock.writeLock().lock();

                        // just store new array
                        updaterParameters.get().setParameters(array);
                        updaterParameters.get().setTimeReceived(System.currentTimeMillis());
                    } finally {
                        updaterParamsLock.writeLock().unlock();
                    }
                }
            });

            // listener for updater params requests
            transport.addRequestConsumer(UpdaterParametersRequest.class, new Consumer<UpdaterParametersRequest>() {
                @Override
                public void accept(UpdaterParametersRequest updaterParametersRequest) throws Exception {
                    // master mode physically can't have own updater parameters, so we're acting as proxy here

                    // we're not requesting updater params if
                    if (!gotFinalState.get()) {

                        // trying to get updaters from root downstreams, excluding original message sender
                        UpdaterParametersMessage updaterParams = transport.sendMessageBlocking(new UpdaterParametersRequest(), transport.getRandomDownstreamFrom(transport.getRootId(), updaterParametersRequest.getOriginatorId()));
                        val uParams = updaterParams.getPayload();

                        try {
                            updaterParamsLock.writeLock().lock();

                            if (updaterParameters.get() == null) {
                                updaterParameters.set(new UpdaterParametersHolder(uParams, System.currentTimeMillis(), false));
                            } else
                                updaterParameters.get().setParameters(uParams);

                        } finally {
                            updaterParamsLock.writeLock().unlock();
                        }
                    }

                    try {
                        updaterParamsLock.readLock().lock();

                        // send updater parameters somewhere
                        log.info("Trying to send back Updater parameters...");
                        val msg = new UpdaterParametersMessage(java.util.UUID.randomUUID().toString(), updaterParameters.get().getParameters());
                        msg.setRequestId(updaterParametersRequest.getRequestId());
                        transport.sendMessage(msg, updaterParametersRequest.getOriginatorId());
                    } finally {
                        updaterParamsLock.readLock().unlock();
                    }
                }
            });
        } else {
            // in case of regular
            transport.addRequestConsumer(UpdaterParametersRequest.class, new Consumer<UpdaterParametersRequest>() {
                @Override
                public void accept(UpdaterParametersRequest updaterParametersRequest) throws Exception {
                    // master mode physically can't have updater parameters
                    log.info("Trying to send back Updater parameters...");
                    if (updaterParametersProvider == null) {
                        log.warn("UpdaterParametersProvider wasn't set!");
                        val msg = new UpdaterParametersMessage(java.util.UUID.randomUUID().toString(), null);
                        msg.setRequestId(updaterParametersRequest.getRequestId());
                        transport.sendMessage(msg, updaterParametersRequest.getOriginatorId());
                    } else {
                        // send updater parameters back
                        val msg = new UpdaterParametersMessage(java.util.UUID.randomUUID().toString(), updaterParametersProvider.getUpdaterParameters());
                        msg.setRequestId(updaterParametersRequest.getRequestId());
                        transport.sendMessage(msg, updaterParametersRequest.getOriginatorId());
                    }
                }
            });
        }

        // this flow will be providing INDArray messages
        disposable = Flowable.fromPublisher(transport.incomingPublisher()).subscribe(message -> {
            /**
             * We process messages here. Messages are either contain INDArrays, say, as gradients update, or as  model parameters.
             */
            if (message instanceof GradientsUpdateMessage) {
                // it's possible to get updates messages BEFORE model was properly initalized
                if (updatesSubscribers.isEmpty()) {
                    log.info("Storing GradientsUpdateMessage into backlog queue...");
                    updatesQueue.add(message.getPayload());
                } else {
                    log.info("Propagating GradientsUpdateMessage to subscribers: [{}]", updatesSubscribers.size());
                    updatesSubscribers.forEach(s -> s.onNext(message.getPayload()));
                }
            } else
                throw new UnsupportedOperationException("Unknown message received: [" + message.getClass().getCanonicalName() + "]");
        });

        // we start transport only once we're ready
        if (this.masterMode)
            transport.launchAsMaster();
        else {
            transport.launch();
        }

        // instance can be stopped now
        stopLock.set(false);

        launchLock.set(true);
    }

    /**
     * This method stops parameter server
     */
    public synchronized void shutdown() {
        if (stopLock.get())
            return;


        // shutting down underlying transport
        transport.shutdown();

        // disposing INDArray flow
        disposable.dispose();

        updaterParamsSubscribers.clear();
        modelParamsSubsribers.clear();
        updatesSubscribers.clear();
        updatesQueue.clear();

        // state that we're done
        launchLock.set(false);

        stopLock.set(true);
    }

    public void sendUpdate(@NonNull INDArray array, int iteration) {
        try {
            //transport.outgoingConsumer().accept(new GradientsUpdateMessage(java.util.UUID.randomUUID().toString(), array));
            val msg = new GradientsUpdateMessage(java.util.UUID.randomUUID().toString(), array);
            msg.setOriginatorId(transport.id());
            msg.setIteration(iteration);
            transport.propagateMessage(msg, PropagationMode.BOTH_WAYS);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method sends gradient updates to the cluster
     */
    public void sendUpdate(@NonNull INDArray array) {
        sendUpdate(array, 0);
    }

    /**
     * This method returns updates received from network
     * @return
     */
    public Collection<INDArray> getUpdates() {
        // just drain stuff from the queue
        val list = new ArrayList<INDArray>();
        updatesQueue.drainTo(list);
        return list;
    }
}
