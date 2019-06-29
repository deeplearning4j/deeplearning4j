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

package org.deeplearning4j.rl4j.observation.transforms;

import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.pooling.CircularFifoObservationPool;
import org.deeplearning4j.rl4j.observation.pooling.ConcatPoolContentAssembler;
import org.deeplearning4j.rl4j.observation.pooling.ObservationPool;
import org.deeplearning4j.rl4j.observation.SimpleObservation;
import org.deeplearning4j.rl4j.observation.VoidObservation;
import org.deeplearning4j.rl4j.observation.pooling.PoolContentAssembler;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * The PoolingTransform will accumulate the observation it receives and will assemble the elements in the pool into the resulting observation. <br>
 * There are two special cases:
 *    1) A VoidObservation is returned when the input is a VoidObservation.
 *    2) When the pool is not ready (i.e. has not yet filled), a VoidOperation is returned.
 * <br>
 * The PoolingTransform requires two sub components: <br>
 *    1) The ObservationPool that supervises what and how input observations are kept. (ex.: Circular FIFO, trailing min/max/avg, etc...)
 *       The default is a Circular FIFO.
 *    2) The PoolContentAssembler that will assemble the pool content into the resulting Observation. (ex.: stacked along a dimention, squashed into a single observation, etc...)
 *       The default is stacking along the dimension 0.
 *
 * @author Alexandre Boulanger
 */
public class PoolingTransform extends PassthroughTransform {

    private final ObservationPool observationPool;
    private final PoolContentAssembler poolContentAssembler;

    protected PoolingTransform(Builder builder)
    {
        observationPool = builder.observationPool;
        poolContentAssembler = builder.poolContentAssembler;
    }

    @Override
    protected Observation handle(Observation input)
    {
        // Do nothing if input is VoidObservation
        if(input instanceof VoidObservation) {
            return input;
        }

        // Add the input, but return VoidObservation if the pool is not ready.
        observationPool.add(input.toNDArray());
        if(!observationPool.isReady()) {
            return VoidObservation.getInstance();
        }

        INDArray result = poolContentAssembler.assemble(observationPool.get());
        return new SimpleObservation(result);
    }

    @Override
    protected boolean getIsReady() {
        return observationPool.isReady();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private ObservationPool observationPool;
        private PoolContentAssembler poolContentAssembler;

        public Builder observablePool(ObservationPool observationPool) {
            this.observationPool = observationPool;
            return this;
        }

        public Builder poolContentAssembler(PoolContentAssembler poolContentAssembler) {
            this.poolContentAssembler = poolContentAssembler;
            return this;
        }

        public PoolingTransform build() {
            if(observationPool == null) {
                observationPool = new CircularFifoObservationPool();
            }

            if(poolContentAssembler == null) {
                poolContentAssembler = new ConcatPoolContentAssembler();
            }

            return new PoolingTransform(this);
        }
    }
}
