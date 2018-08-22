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

package org.deeplearning4j.parallelism;


import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.inference.LoadBalanceMode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * This ParallelInference implementation provides inference functionality without launching additional threads, so inference happens in the calling thread.
 *
 * To instantiate this implementation one should use InferenceMode.INPLACE in ParallelInference.Builder
 *
 * PLEASE NOTE: This implementation does not create additional threads
 * PLEASE NOTE: This implementation uses shared parameters for models on per-device basis
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class InplaceParallelInference extends ParallelInference {
    protected List<ModelHolder> holders = new CopyOnWriteArrayList<>();
    protected ModelSelector selector = new ModelSelector();

    protected final Object locker = new Object();

    @Override
    protected void init() {
        for (int e = 0; e < Nd4j.getAffinityManager().getNumberOfDevices(); e++) {
            val h = ModelHolder.builder()
                    .sourceModel(model)
                    .workers(workers)
                    .loadBalanceMode(loadBalanceMode)
                    .targetDeviceId(e)
                    .rootDevice(e == Nd4j.getAffinityManager().getDeviceForCurrentThread().intValue())
                    .build();
            h.init();

            // adding for simplified access
            holders.add(h);

            // and adding it to actual
            selector.addModelHolder(e, h);
        }
    }

    @Override
    public synchronized void updateModel(@NonNull Model model) {
        for (val h:holders)
            h.updateModel(model);
    }

    @Override
    protected synchronized Model[] getCurrentModelsFromWorkers() {
        val models = new Model[holders.size()];
        int cnt = 0;
        for (val h:holders) {
            models[cnt++] = h.sourceModel;
        }

        return models;
    }

    @Override
    public INDArray[] output(INDArray[] input, INDArray[] inputMasks) {
        return selector.output(input, inputMasks);
    }


    protected static class ModelSelector {
        // this map stores collection of shared
        protected Map<Integer, ModelHolder> map = new HashMap<>();

        protected final LoadBalanceMode loadBalanceMode;

        public ModelSelector() {
            this(LoadBalanceMode.ROUND_ROBIN);
        }

        public ModelSelector(LoadBalanceMode loadBalanceMode) {
            this.loadBalanceMode = loadBalanceMode;
        }

        protected void addModelHolder(@NonNull Integer device, @NonNull ModelHolder holder) {
            map.put(device, holder);
        }

        public ModelHolder getModelForThread(long threadId) {
            // first of all we get mapped device for this thread
            val device = Nd4j.getAffinityManager().getDeviceForThread(threadId);

            // each device has it's own queue
            val q = map.get(device);

            // and we're returning holder right away
            return q;
        }

        public INDArray[] output(INDArray[] input, INDArray[] inputMasks) {
            return getModelForThisThread().output(input, inputMasks);
        }

        public ModelHolder getModelForThisThread() {
            return getModelForThread(Thread.currentThread().getId());
        }
    }

    @NoArgsConstructor
    @AllArgsConstructor
    @lombok.Builder
    protected static class ModelHolder {
        protected Model sourceModel;
        @lombok.Builder.Default protected int workers = 4;
        @lombok.Builder.Default protected List<Model> replicas = new ArrayList<>();
        @lombok.Builder.Default protected boolean rootDevice = true;
        @lombok.Builder.Default protected LoadBalanceMode loadBalanceMode = LoadBalanceMode.ROUND_ROBIN;
        protected int targetDeviceId;

        protected final AtomicLong position = new AtomicLong(0);
        protected final ReentrantReadWriteLock modelLock = new ReentrantReadWriteLock();

        // this queue is used in FIFO mode
        protected final BlockingQueue<Model> queue = new LinkedBlockingQueue<>();

        @lombok.Builder.Default protected transient boolean isCG = false;
        @lombok.Builder.Default protected transient boolean isMLN = false;


        protected synchronized void init() {
            if (workers < 1)
                throw new ND4JIllegalStateException("Workers must be positive value");

            replicas.clear();

            isCG = sourceModel instanceof ComputationGraph;
            isMLN = sourceModel instanceof MultiLayerNetwork;

            // we clone params only if we're not on the same device
            val params = rootDevice ? sourceModel.params() : sourceModel.params().unsafeDuplication(true);

            // and moving it to specified device (only if NOT root
            if (!rootDevice)
                Nd4j.getAffinityManager().replicateToDevice(targetDeviceId, params);

            for (int e = 0; e < workers; e++) {
                if (sourceModel instanceof ComputationGraph) {
                    // building configuration with shared parameters
                    val model = new ComputationGraph(ComputationGraphConfiguration.fromJson(((ComputationGraph) sourceModel).getConfiguration().toJson()));
                    model.init(params, false);
                    Nd4j.getExecutioner().commit();

                    // storing model for future reuse
                    replicas.add(model);

                    if (loadBalanceMode == LoadBalanceMode.FIFO)
                        queue.add(model);
                } else if (sourceModel instanceof MultiLayerNetwork) {
                    val model = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(((MultiLayerNetwork) sourceModel).getLayerWiseConfigurations().toJson()));
                    model.init(params, false);
                    Nd4j.getExecutioner().commit();

                    replicas.add(model);

                    if (loadBalanceMode == LoadBalanceMode.FIFO)
                        queue.add(model);
                }
            }
        }


        protected Model acquireModel() throws InterruptedException {
            try {
                modelLock.readLock().lock();

                switch (loadBalanceMode) {
                    case FIFO: {
                            return queue.take();
                        }
                    case ROUND_ROBIN:
                        return replicas.get((int) (position.getAndIncrement() % replicas.size()));
                    default:
                        throw new ND4JIllegalStateException("Unknown LoadBalanceMode was specified: [" + loadBalanceMode + "]");
                }
            } finally {
                modelLock.readLock().unlock();
            }
        }

        protected void releaseModel(Model model) {
            try {
                modelLock.readLock().lock();

                switch (loadBalanceMode) {
                    case FIFO:
                        queue.add(model);
                        break;
                    case ROUND_ROBIN:
                        break;
                    default:
                        throw new ND4JIllegalStateException("Unknown LoadBalanceMode was specified: [" + loadBalanceMode + "]");
                }
            } finally {
                modelLock.readLock().unlock();
            }
        }

        protected INDArray[] output(INDArray[] input, INDArray[] inputMasks) {
            try {
                modelLock.readLock().lock();
                if (isCG) {
                    // acquiring model from pool
                    val model = acquireModel();

                    // doing inference
                    val output = ((ComputationGraph) model).output(false, input, inputMasks);

                    // releasing model
                    releaseModel(model);
                    return output;
                } else if (isMLN) {
                    if (input.length > 1 || (inputMasks != null && inputMasks.length > 1))
                        throw new ND4JIllegalStateException("MultilayerNetwork can't have multiple inputs");

                    val model = acquireModel();
                    val result = ((MultiLayerNetwork) model).output(input[0], false, inputMasks[0], null);
                    releaseModel(model);
                    return new INDArray[]{result};
                } else
                    throw new UnsupportedOperationException();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            } finally {
                modelLock.readLock().unlock();
            }
        }

        protected void updateModel(@NonNull Model model) {
            try {
                modelLock.writeLock().lock();

                this.sourceModel = model;

                init();
            } finally {
                modelLock.writeLock().unlock();
            }
        }

    }
}
