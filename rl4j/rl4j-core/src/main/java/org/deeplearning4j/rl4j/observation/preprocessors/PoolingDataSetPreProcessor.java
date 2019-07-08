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

package org.deeplearning4j.rl4j.observation.preprocessors;

import org.deeplearning4j.rl4j.observation.preprocessors.pooling.ChannelStackPoolContentAssembler;
import org.deeplearning4j.rl4j.observation.preprocessors.pooling.PoolContentAssembler;
import org.deeplearning4j.rl4j.observation.preprocessors.pooling.CircularFifoObservationPool;
import org.deeplearning4j.rl4j.observation.preprocessors.pooling.ObservationPool;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;

/**
 * The PoolingDataSetPreProcessor will accumulate features from incoming DataSets and will assemble its content
 * into a DataSet containing a single example.
 *
 * There are two special cases:
 *    1) preProcess will return without doing anything if the input DataSet is empty
 *    2) When the pool has not yet filled, the data from the incoming DataSet is stored in the pool but the DataSet is emptied
 *       on exit.
 * <br>
 * The PoolingDataSetPreProcessor requires two sub components: <br>
 *    1) The ObservationPool that supervises what and how input observations are kept. (ex.: Circular FIFO, trailing min/max/avg, etc...)
 *       The default is a Circular FIFO.
 *    2) The PoolContentAssembler that will assemble the pool content into a resulting single INDArray. (ex.: stacked along a dimention, squashed into a single observation, etc...)
 *       The default is stacking along the dimension 0.
 *
 * @author Alexandre Boulanger
 */
public class PoolingDataSetPreProcessor extends ResettableDataSetPreProcessor {
    private final ObservationPool observationPool;
    private final PoolContentAssembler poolContentAssembler;

    protected PoolingDataSetPreProcessor(PoolingDataSetPreProcessor.Builder builder)
    {
        observationPool = builder.observationPool;
        poolContentAssembler = builder.poolContentAssembler;
    }

    /**
     * Note: preProcess will empty the processed dataset if the pool has not filled. Empty datasets should ignored by the
     * Policy/Learning class and other DataSetPreProcessors
     *
     * @param dataSet
     */
    @Override
    public void preProcess(DataSet dataSet) {
        Preconditions.checkNotNull(dataSet, "Encountered null dataSet");

        if(dataSet.isEmpty()) {
            return;
        }

        Preconditions.checkArgument(dataSet.numExamples() == 1, "Pooling datasets conatining more than one example is not supported");

        // store a duplicate in the pool
        observationPool.add(dataSet.getFeatures().slice(0, 0).dup());
        if(!observationPool.isAtFullCapacity()) {
            dataSet.setFeatures(null);
            return;
        }

        INDArray result = poolContentAssembler.assemble(observationPool.get());

        // return a DataSet containing only 1 example (the result)
        long[] resultShape = result.shape();
        long[] newShape = new long[resultShape.length + 1];
        newShape[0] = 1;
        System.arraycopy(resultShape, 0, newShape, 1, resultShape.length);

        dataSet.setFeatures(result.reshape(newShape));
    }

    public static PoolingDataSetPreProcessor.Builder builder() {
        return new PoolingDataSetPreProcessor.Builder();
    }

    @Override
    public void reset() {
        observationPool.reset();
    }

    public static class Builder {
        private ObservationPool observationPool;
        private PoolContentAssembler poolContentAssembler;

        /**
         * Default is CircularFifoObservationPool
         */
        public PoolingDataSetPreProcessor.Builder observablePool(ObservationPool observationPool) {
            this.observationPool = observationPool;
            return this;
        }

        /**
         * Default is ChannelStackPoolContentAssembler
         */
        public PoolingDataSetPreProcessor.Builder poolContentAssembler(PoolContentAssembler poolContentAssembler) {
            this.poolContentAssembler = poolContentAssembler;
            return this;
        }

        public PoolingDataSetPreProcessor build() {
            if(observationPool == null) {
                observationPool = new CircularFifoObservationPool();
            }

            if(poolContentAssembler == null) {
                poolContentAssembler = new ChannelStackPoolContentAssembler();
            }

            return new PoolingDataSetPreProcessor(this);
        }
    }

}
