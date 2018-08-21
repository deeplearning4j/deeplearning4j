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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * This ParallelInference implementation provides inference functionality without launching additional threads, so inference happens in the calling thread
 * @author raver119@gmail.com
 */
@Slf4j
public class InplaceParallelInference extends ParallelInference {
    protected List<ModelHolder> holders = new CopyOnWriteArrayList<>();
    protected ModelSelector selector = new ModelSelector();
    protected boolean isCG = false;
    protected boolean isMLN = false;

    protected final Object locker = new Object();

    @Override
    protected void init() {
        if (model instanceof ComputationGraph)
            isCG = true;
        else if (model instanceof MultiLayerNetwork)
            isMLN = true;
        else
            throw new ND4JIllegalStateException("Unknown model was passed into ParallelInference: [" + model.getClass().getCanonicalName() + "]");

        for (int e = 0; e < Nd4j.getAffinityManager().getNumberOfDevices(); e++) {
            val h = ModelHolder.builder()
                    .sourceModel(model)
                    .workers(workers)
                    .build();
            h.init();

            // adding for simplified access
            holders.add(h);

            //
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
            models[cnt++] = h.getReplicatedModel();
        }

        return models;
    }

    @Override
    public INDArray[] output(INDArray[] input, INDArray[] inputMasks){
        if (isCG) {
            return ((ComputationGraph) selector.getModelForThisThread()).output(false, input, inputMasks);
        } else if (isMLN) {
            if (input.length > 1 || inputMasks.length > 1)
                throw new ND4JIllegalStateException("MultilayerNetwork can't have multiple inputs");

            val result = ((MultiLayerNetwork) selector.getModelForThisThread()).output(input[0], false, inputMasks[0], null);
            return new INDArray[] {result};
        } else
            throw new UnsupportedOperationException();
    }


    public static class ModelSelector {
        // this map stores collection of shared
        protected Map<Integer, ModelHolder> map = new HashMap<>();

        protected void addModelHolder(@NonNull Integer device, @NonNull ModelHolder holder) {
            map.put(device, holder);
        }

        public Model getModelForThread(long threadId) {
            // first of all we get mapped device for this thread
            val device = Nd4j.getAffinityManager().getDeviceForThread(threadId);

            // each device has it's own queue
            val q = map.get(device);

            // and we're returning model right away
            return q.getReplicatedModel();
        }

        public Model getModelForThisThread() {
            return getModelForThread(Thread.currentThread().getId());
        }
    }

    @NoArgsConstructor
    @AllArgsConstructor
    @lombok.Builder
    public static class ModelHolder {
        protected Model sourceModel;
        @lombok.Builder.Default protected int workers = 4;
        @lombok.Builder.Default protected List<Model> replicas = new ArrayList<>();
        @lombok.Builder.Default protected boolean rootDevice = true;
         protected final AtomicLong position = new AtomicLong(0);

        protected final ReentrantReadWriteLock modelLock = new ReentrantReadWriteLock();

        protected synchronized void init() {
            if (workers < 1)
                throw new ND4JIllegalStateException("Workers must be positive value");

            replicas.clear();

            // we clone params only if we're not on the same device
            val params = rootDevice ? sourceModel.params() : sourceModel.params().unsafeDuplication(true);
            for (int e = 0; e < workers; e++) {
                if (sourceModel instanceof ComputationGraph) {
                    // building configuration with shared parameters
                    val model = new ComputationGraph(ComputationGraphConfiguration.fromJson(((ComputationGraph) sourceModel).getConfiguration().toJson()));
                    model.init(params, false);
                    Nd4j.getExecutioner().commit();

                    // storing model for future reuse
                    replicas.add(model);
                } else if (sourceModel instanceof MultiLayerNetwork) {
                    val model = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(((MultiLayerNetwork) sourceModel).getLayerWiseConfigurations().toJson()));
                    model.init(params, false);
                    Nd4j.getExecutioner().commit();

                    replicas.add(model);
                }
            }
        }

        protected Model getReplicatedModel() {
            try {
                modelLock.readLock().lock();

                return replicas.get((int) (position.getAndIncrement() % replicas.size()));
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
