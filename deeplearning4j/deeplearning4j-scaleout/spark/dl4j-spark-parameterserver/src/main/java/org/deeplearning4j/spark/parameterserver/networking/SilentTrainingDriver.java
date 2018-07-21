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

package org.deeplearning4j.spark.parameterserver.networking;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.solvers.accumulation.FancyBlockingQueue;
import org.deeplearning4j.optimize.solvers.accumulation.GradientsAccumulator;
import org.deeplearning4j.spark.parameterserver.networking.messages.SilentUpdatesMessage;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.ThresholdCompression;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.logic.Storage;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.messages.VoidAggregation;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.transport.Transport;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This TrainingDriver implementation is suited ONLY for Spark Master, and handles application & redistribution of incoming encoded messages across distributed network
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class SilentTrainingDriver implements TrainingDriver<SilentUpdatesMessage> {
    protected transient INDArray params;
    protected transient INDArray updates;
    protected transient StepFunction stepFunction;

    protected transient GradientsAccumulator accumulator;

    protected transient VoidConfiguration voidConfiguration;
    protected transient Transport transport;
    protected transient AtomicLong updatesCount;
    protected transient AtomicBoolean hasSomething;

    protected transient AtomicBoolean bypassMode = new AtomicBoolean(false);

    protected transient AtomicLong denseCounter = new AtomicLong(0);
    protected transient AtomicLong sparseCounter = new AtomicLong(0);

    /*
        We use this buffer to provide double buffering for incoming messages.
        So we store incoming messages right here, and apply them as time comes
     */
    protected transient BlockingQueue<INDArray> updatesBuffer;

    // these 2 are not used here
    protected transient Storage storage;
    protected transient Clipboard clipboard;


    public SilentTrainingDriver(@NonNull GradientsAccumulator accumulator) {
        log.info("Creating TrainingDriver for worker...");
        this.accumulator = accumulator;
        this.updatesCount = new AtomicLong(0);

        // TODO: make this configurable
        this.updatesBuffer = new FancyBlockingQueue<>(new LinkedBlockingQueue<>(1024));

        // FBQ will guarantee that all workers using given queue will be applying the same updates in the same order
        this.accumulator.setExternalSource(updatesBuffer);
    }

    public SilentTrainingDriver(@NonNull INDArray params, @NonNull StepFunction stepFunction) {
        log.info("Creating TrainingDriver for master...");
        log.info("Params at Master BEFORE: {}", params.meanNumber().doubleValue());
        this.params = params;
        this.stepFunction = stepFunction;
        this.updatesCount = new AtomicLong(0);

        this.hasSomething = new AtomicBoolean(false);

        // updates are always the same size as params
        try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            this.updates = Nd4j.create(params.shape(), params.ordering());
        }
    }

    /**
     * This method is viable only at Spark Workers, Master node will always have empty buffer here by design
     * @return
     */
    public BlockingQueue<INDArray> getUpdatesBuffer() {
        return updatesBuffer;
    }

    @Override
    public void init(@NonNull VoidConfiguration voidConfiguration, @NonNull Transport transport, Storage storage,
                    Clipboard clipboard) {
        this.voidConfiguration = voidConfiguration;
        this.transport = transport;
    }

    public void bypassMode(boolean reallyBypass) {
        bypassMode.set(reallyBypass);

        // if TrainingDriver is temporary disabled - remove existing messages from queue
        if (reallyBypass) {
            updatesBuffer.clear();
        }
    }

    @Override
    public void startTraining(SilentUpdatesMessage message) {
        System.out.println(">>> SilentTrainingDriver: processing message");
        /*
            this method will be invoked on master, and will do 2 things:
            1) silently update params via given StepFunction
            2) propagate this message to everyone
        
            on workers, it just enqueues updates into the FancyBlockingQueue
         */
        // if accumulator is defined, we're working at Worker level, so it's not our problem what happens inside
        if (accumulator != null) {
            if (message.getOriginatorId() == transport.getOwnOriginatorId()) {
                //log.info("Skipping since originators match");
                return;
            } ;

            /*
                we're just putting messages here. if thread gets blocked - messages won't be arriving,
                enforcing periodic messages retransmission from other nodes, so we should be all fine
              */

            try {
                if (!bypassMode.get()) {
                    updatesBuffer.put(message.getUpdates());
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            //accumulator.receiveUpdate(message.getUpdates());
        } else if (params != null && stepFunction != null) {
            // master invokes everything, since that's Silent Worker approach: we want master to be always up-to-date
            synchronized (this) {
                // threshold decoder is inplace & fast
                int encoding = message.getUpdates().data().getInt(3);
                if (encoding == ThresholdCompression.FLEXIBLE_ENCODING) {
                    Nd4j.getExecutioner().thresholdDecode(message.getUpdates(), updates);
                    sparseCounter.incrementAndGet();
                } else if (encoding == ThresholdCompression.BITMAP_ENCODING) {
                    Nd4j.getExecutioner().bitmapDecode(message.getUpdates(), updates);
                    denseCounter.incrementAndGet();
                } else
                    throw new DL4JInvalidConfigException("Unknown compression header received: " + encoding);

                /*
                if ((sparseCounter.get() + denseCounter.get()) % 100 == 0) {
                    log.info("Sparse/Dense ratio: {}", String.format("%.2f", (sparseCounter.get() +1) / (double) (denseCounter.get() + 1)));
                }
                */


                // this simple flag shows that we have something not applied, will be used at finishTraining() method
                hasSomething.set(true);

                // we apply updates every X iterations, and we don't really need X to be small here
                if (updatesCount.incrementAndGet() % Math.max(transport.numberOfKnownClients(), 5) == 0) {
                    System.out.println("<<< Applied update to parameters: " + params.hashCode() + " - instance " + this);
                    stepFunction.step(params, updates);

                    // once accumulated updates are applied - reset storage, and wait for other messsages
                    Nd4j.getMemoryManager().memset(updates);
                    hasSomething.set(false);
                }
            }

            // we should echo this message to everyone but this shard, but only if there's > 1 shard/client available
            if (transport.numberOfKnownClients() > 1) {
                //log.info("Resending message, skipping {}", message.getOriginatorId());
                transport.sendMessageToAllClients(message, message.getOriginatorId(), transport.getOwnOriginatorId());
            } // else log.info("No known Clients so far");
        } else
            throw new DL4JInvalidConfigException("Neither GradientsAccumulator or StepFunction is defined!");
    }

    @Override
    public void pickTraining(SilentUpdatesMessage message) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void aggregationFinished(VoidAggregation aggregation) {
        throw new UnsupportedOperationException();
    }

    /**
     * This method is used on Master only, applies buffered updates to params
     *
     * @param originatorId
     * @param taskId
     */
    @Override
    public void finishTraining(long originatorId, long taskId) {
        // on Master thread we'll be applying final gradients

        if (params != null && stepFunction != null) {
            if (hasSomething.get()) {
                stepFunction.step(params, updates);
                //Nd4j.getMemoryManager().memset(updates);
                updates.assign(0.0);
            }
        }

    }

    @Override
    public void addCompletionHook(long originatorId, long frameId, long messageId) {
        // no-op
        throw new UnsupportedOperationException();
    }

    @Override
    public String targetMessageClass() {
        return SilentUpdatesMessage.class.getSimpleName();
    }
}
