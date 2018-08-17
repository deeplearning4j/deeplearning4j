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
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
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

import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 *
 */
@Slf4j
public final class ModelParameterServer {
    private final Transport transport;

    // TODO: we need better capacity here, it should scale properly
    private final BlockingQueue<INDArray> updatesQueue = new LinkedBlockingQueue<>(4096);

    private final boolean masterMode;

    protected final VoidConfiguration configuration;

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
    public ModelParameterServer(VoidConfiguration configuration, Transport transport, boolean isMasterNode) {
        this.transport = transport;
        this.masterMode = isMasterNode;
        this.configuration = configuration;
    }

    // this flag is true once mps is launched
    private final AtomicBoolean launchLock = new AtomicBoolean(false);
    private final AtomicBoolean stopLock = new AtomicBoolean(false);

    private Disposable disposable;

    /**
     * This method starts parameter server
     */
    public synchronized void launch() {
        if (launchLock.get())
            return;

        transport.setRestartCallback(new RestartCallback() {
            @Override
            public void call(HandshakeResponse response) {
                // upon restart command we'll request current parameters from the current upstream (without any propagation
                try {
                    ModelParametersMessage modelParams = transport.sendMessageBlocking(new ModelParametersRequest(), transport.getUpstreamId());
                    val mParams = modelParams.getPayload();

                    // updater parameters are optional, it's possible to have models without updater parameters (i.e. SGD)
                    UpdaterParametersMessage updaterParams = transport.sendMessageBlocking(new UpdaterParametersRequest(), transport.getUpstreamId());
                    val uParams = updaterParams.getPayload();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        });

        // listener for model params requests
        transport.addRequestConsumer(ModelParametersRequest.class, new Consumer<ModelParametersRequest>() {
            @Override
            public void accept(ModelParametersRequest modelParametersRequest) throws Exception {
                // send model parameters somewhere
            }
        });

        // listener for updater params requests
        transport.addRequestConsumer(UpdaterParametersRequest.class, new Consumer<UpdaterParametersRequest>() {
            @Override
            public void accept(UpdaterParametersRequest updaterParametersRequest) throws Exception {
                // send updater parameters somewhere
            }
        });

        // this flow will be providing INDArray messages
        disposable = Flowable.fromPublisher(transport.incomingPublisher()).subscribe(message -> {
            /**
             * We process messages here. Messages are either contain INDArrays, say, as gradients update, or as  model parameters.
             */
            if (message instanceof GradientsUpdateMessage) {
                updatesQueue.add(message.getPayload());
            } else
                throw new UnsupportedOperationException("Unknown message received: [" + message.getClass().getCanonicalName() + "]");
        });

        // we start transport only once we're ready
        if (this.masterMode)
            transport.launchAsMaster();
        else {
            transport.launch();
        }

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

        stopLock.set(true);
    }

    /**
     * This method sends gradient updates to the cluster
     */
    public void sendUpdate(@NonNull INDArray array) {
        try {
            //transport.outgoingConsumer().accept(new GradientsUpdateMessage(java.util.UUID.randomUUID().toString(), array));
            val msg = new GradientsUpdateMessage(java.util.UUID.randomUUID().toString(), array);
            msg.setOriginatorId(transport.id());
            transport.propagateMessage(msg, PropagationMode.BOTH_WAYS);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
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
