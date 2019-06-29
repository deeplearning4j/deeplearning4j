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

package org.deeplearning4j.rl4j.observation.pooling;

import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * CircularFifoObservationPool is used with the PoolingTransform. The CircularFifoObservationPool will become ready
 * when <i>poolSize</i> elements have been added. After which the pool will return the most recent <i>poolSize</i> elements.
 *
 * @author Alexandre Boulanger
 */
public class CircularFifoObservationPool implements ObservationPool {
    private static final int DEFAULT_POOL_SIZE = 4;

    private final CircularFifoQueue<INDArray> queue;

    private CircularFifoObservationPool(Builder builder) {
        queue = new CircularFifoQueue<>(builder.poolSize);
    }

    public CircularFifoObservationPool()
    {
        this(DEFAULT_POOL_SIZE);
    }

    public CircularFifoObservationPool(int poolSize)
    {
        Preconditions.checkArgument(poolSize > 0, "The pool size must be at least 1, got %s", poolSize);
        queue = new CircularFifoQueue<>(poolSize);
    }

    /**
     * Add an element to the pool, if this addition would make the pool to overflow, the added element replaces the oldest one.
     * @param elem
     */
    public void add(INDArray elem) {
        queue.add(elem);
    }

    /**
     * @return The content of the pool, returned in order from oldest to newest.
     */
    public INDArray[] get() {
        int size = queue.size();
        INDArray[] array = new INDArray[size];
        for (int i = 0; i < size; ++i) {
            array[i] = queue.get(i).castTo(Nd4j.dataType());
        }
        return array;
    }

    /**
     * @return True if the pool is full.
     */
    public boolean isReady() {
        return queue.isAtFullCapacity();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private int poolSize = DEFAULT_POOL_SIZE;

        public Builder poolSize(int poolSize) {
            this.poolSize = poolSize;
            return this;
        }

        public CircularFifoObservationPool build() {
            return new CircularFifoObservationPool(this);
        }
    }
}
