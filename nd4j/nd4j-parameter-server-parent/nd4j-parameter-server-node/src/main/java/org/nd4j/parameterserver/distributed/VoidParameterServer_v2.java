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

package org.nd4j.parameterserver.distributed;

import io.reactivex.Flowable;
import io.reactivex.functions.Consumer;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.messages.v2.INDArrayMessage;
import org.nd4j.parameterserver.distributed.transport.Transport_v2;
import org.reactivestreams.Subscriber;

import java.util.Collection;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 *
 */
@Slf4j
public final class VoidParameterServer_v2 {
    private final Transport_v2 transport;

    // TODO: we need better capacity here, it should scale properly
    private final BlockingQueue<INDArray> updatesQueue = new LinkedBlockingQueue<>(4096);

    public VoidParameterServer_v2(@NonNull Transport_v2 transport) {
        this.transport = transport;
    }

    /**
     * This method starts parameter server
     */
    public synchronized void launch() {
        Flowable.fromPublisher(transport.incomingPublisher()).subscribe(new Consumer<VoidMessage>() {
            @Override
            public void accept(VoidMessage message) throws Exception {
                /**
                 * We process messages here. Messages are either contain INDArrays, say, as gradients update, or as  model parameters.
                 *
                 */
                if (message instanceof INDArrayMessage)
                    updatesQueue.add(((INDArrayMessage) message).exec(null));
                else
                    message.processMessage();
            }
        });
    }

    /**
     * This method stops parameter server
     */
    public synchronized void shutdown() {

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
        return null;
    }
}
