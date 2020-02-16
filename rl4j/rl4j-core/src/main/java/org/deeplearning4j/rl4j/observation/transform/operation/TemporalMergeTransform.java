/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
package org.deeplearning4j.rl4j.observation.transform.operation;

import org.datavec.api.transform.Operation;
import org.deeplearning4j.rl4j.helper.INDArrayHelper;
import org.deeplearning4j.rl4j.observation.transform.ResettableOperation;
import org.deeplearning4j.rl4j.observation.transform.operation.temporalmerge.CircularFifoStore;
import org.deeplearning4j.rl4j.observation.transform.operation.temporalmerge.TemporalStackAssembler;
import org.deeplearning4j.rl4j.observation.transform.operation.temporalmerge.TemporalMergeAssembler;
import org.deeplearning4j.rl4j.observation.transform.operation.temporalmerge.TemporalMergeElementStore;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * The TemporalMergeTransform will accumulate features from incoming INDArrays and will assemble its content
 * into a new INDArray containing a single example.
 *
 * This is used in scenarios where motion in an important element.
 *
 * There is a special case:
 *    * When the store is not full (not ready), the data from the incoming INDArray is stored but null is returned (will be interpreted as a skipped observation)
 * <br>
 * The TemporalMergeTransform requires two sub components: <br>
 *    1) The {@link TemporalMergeElementStore TemporalMergeElementStore} that supervises what and how input INDArrays are kept. (ex.: Circular FIFO, trailing min/max/avg, etc...)
 *       The default is a Circular FIFO.
 *    2) The {@link TemporalMergeAssembler TemporalMergeAssembler} that will assemble the store content into a resulting single INDArray. (ex.: stacked along a dimension, squashed into a single observation, etc...)
 *       The default is stacking along the dimension 0.
 *
 * @author Alexandre Boulanger
 */
public class TemporalMergeTransform implements Operation<INDArray, INDArray>, ResettableOperation {

    private final TemporalMergeElementStore temporalMergeElementStore;
    private final TemporalMergeAssembler temporalMergeAssembler;
    private final boolean shouldStoreCopy;
    private final boolean isFirstDimenstionBatch;

    private TemporalMergeTransform(Builder builder) {
        this.temporalMergeElementStore = builder.temporalMergeElementStore;
        this.temporalMergeAssembler = builder.temporalMergeAssembler;
        this.shouldStoreCopy = builder.shouldStoreCopy;
        this.isFirstDimenstionBatch = builder.isFirstDimenstionBatch;
    }

    @Override
    public INDArray transform(INDArray input) {
        INDArray element;
        if(isFirstDimenstionBatch) {
            element = input.slice(0, 0);
        }
        else {
            element = input;
        }

        if(shouldStoreCopy) {
            element = element.dup();
        }

        temporalMergeElementStore.add(element);
        if(!temporalMergeElementStore.isReady()) {
            return null;
        }

        INDArray result = temporalMergeAssembler.assemble(temporalMergeElementStore.get());

        return INDArrayHelper.forceCorrectShape(result);
    }

    @Override
    public void reset() {
        temporalMergeElementStore.reset();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private TemporalMergeElementStore temporalMergeElementStore;
        private TemporalMergeAssembler temporalMergeAssembler;
        private boolean shouldStoreCopy = false;
        private boolean isFirstDimenstionBatch = false;

        /**
         * Default is {@link CircularFifoStore CircularFifoStore}
         */
        public Builder elementStore(TemporalMergeElementStore store) {
            this.temporalMergeElementStore = store;
            return this;
        }

        /**
         * Default is {@link TemporalStackAssembler TemporalStackAssembler}
         */
        public Builder assembler(TemporalMergeAssembler assembler) {
            this.temporalMergeAssembler = assembler;
            return this;
        }

        /**
         * If true, tell the TemporalMergeTransform to store copies of incoming INDArrays.
         * (To prevent later in-place changes to a stored INDArray from changing what has been stored)
         *
         * Default is false
         */
        public Builder shouldStoreCopy(boolean shouldStoreCopy) {
            this.shouldStoreCopy = shouldStoreCopy;
            return this;
        }

        /**
         * If true, tell the TemporalMergeTransform that the first dimension of the input INDArray is the batch count.
         * When this is the case, the TemporalMergeTransform will slice the input like this [batch, height, width] -> [height, width]
         *
         * Default is false
         */
        public Builder isFirstDimenstionBatch(boolean isFirstDimenstionBatch) {
            this.isFirstDimenstionBatch = isFirstDimenstionBatch;
            return this;
        }

        public TemporalMergeTransform build() {
            if(temporalMergeElementStore == null) {
                temporalMergeElementStore = new CircularFifoStore();
            }

            if(temporalMergeAssembler == null) {
                temporalMergeAssembler = new TemporalStackAssembler();
            }

            return new TemporalMergeTransform(this);
        }
    }
}
