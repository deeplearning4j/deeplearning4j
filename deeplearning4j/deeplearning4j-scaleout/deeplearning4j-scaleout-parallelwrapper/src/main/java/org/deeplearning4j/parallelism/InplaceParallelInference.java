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
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * This ParallelInference implementation provides inference functionality without launching additional threads, so inference happens in the calling thread
 * @author raver119@gmail.com
 */
@Slf4j
public class InplaceParallelInference extends ParallelInference {
    protected List<ModelHolder> holders = new CopyOnWriteArrayList<>();
    protected boolean isCG = false;
    protected boolean isMLN = false;

    @Override
    protected void init() {
        if (model instanceof ComputationGraph)
            isCG = true;
        else if (model instanceof MultiLayerNetwork)
            isMLN = true;
        else
            throw new ND4JIllegalStateException("Unknown model was passed into ParallelInference: [" + model.getClass().getCanonicalName() + "]");

        for (int e = 0; e < workers; e++) {
            val h = ModelHolder.builder().sourceModel(model).build();
            h.init();
            holders.add(h);
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
            models[cnt++] = h.replicatedModel;
        }

        return models;
    }

    @Override
    public INDArray[] output(INDArray[] input, INDArray[] inputMasks){
        if (isCG) {

            return new INDArray[0];
        } else if (isMLN) {
            if (input.length > 1 || inputMasks.length > 1)
                throw new ND4JIllegalStateException("MultilayerNetwork can't have multiple inputs");

            val result = ((MultiLayerNetwork) model).output(input[0], false, inputMasks[0], null);
            return new INDArray[] {result};
        } else
            throw new UnsupportedOperationException();
    }


    @NoArgsConstructor
    @AllArgsConstructor
    @lombok.Builder
    public static class ModelSelector {

    }

    @NoArgsConstructor
    @AllArgsConstructor
    @lombok.Builder
    public static class ModelHolder {
        protected Model sourceModel;
        protected Model replicatedModel;
        @lombok.Builder.Default protected boolean rootDevice = true;

        protected final ReentrantReadWriteLock modelLock = new ReentrantReadWriteLock();

        protected synchronized void init() {
            if (rootDevice)
                this.replicatedModel = this.sourceModel;
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
