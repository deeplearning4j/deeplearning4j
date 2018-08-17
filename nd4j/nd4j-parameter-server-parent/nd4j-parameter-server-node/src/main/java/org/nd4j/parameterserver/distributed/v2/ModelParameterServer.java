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
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.parameterserver.distributed.v2.messages.impl.GradientsUpdateMessage;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.params.ModelParametersMessage;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.params.UpdaterParametersMessage;
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

    public ModelParameterServer(@NonNull Transport transport) {
        this.transport = transport;
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

        // this flow will be providing INDArray messages
        disposable = Flowable.fromPublisher(transport.incomingPublisher()).subscribe(message -> {
            /**
             * We process messages here. Messages are either contain INDArrays, say, as gradients update, or as  model parameters.
             */
            if (message instanceof GradientsUpdateMessage) {
                updatesQueue.add(message.getPayload());
            } else if (message instanceof ModelParametersMessage) {
                //
            } else if (message instanceof UpdaterParametersMessage) {
                //
            }
        });

        launchLock.set(true);
    }

    /**
     * This method stops parameter server
     */
    public synchronized void shutdown() {
        if (stopLock.get())
            return;

        // disposing INDArray flow
        disposable.dispose();

        stopLock.set(true);
    }

    /**
     * This method sends gradient updates to the cluster
     */
    public void sendUpdate(INDArray array) {
        try {
            transport.outgoingConsumer().accept(null);
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
