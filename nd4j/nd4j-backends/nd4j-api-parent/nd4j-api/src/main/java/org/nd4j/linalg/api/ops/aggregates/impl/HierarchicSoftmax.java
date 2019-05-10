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

package org.nd4j.linalg.api.ops.aggregates.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.BaseAggregate;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This Op describes HS round for AggregateSkipGram/CBOW Hierarchic Softmax
 *
 * @author raver119@gmail.com
 */
public class HierarchicSoftmax extends BaseAggregate {
    private int vectorLength;

    public HierarchicSoftmax(INDArray syn0, INDArray syn1, INDArray expTable, INDArray neu1e, int code, double lr) {
        arguments.add(syn0);
        arguments.add(syn1);
        arguments.add(expTable);
        arguments.add(neu1e);

        // FIXME: int cast

        indexingArguments.add((int) neu1e.length());
        indexingArguments.add((int) expTable.length());
        indexingArguments.add(code);
        indexingArguments.add(0); // set isInference to false

        realArguments.add(lr);

        this.vectorLength = (int) neu1e.length();
    }

    /**
     * This method returns amount of shared memory required for this specific Aggregate.
     * PLEASE NOTE: this method is especially important for CUDA backend. On CPU backend it might be ignored, depending on Aggregate.
     *
     * @return
     */
    @Override
    public int getSharedMemorySize() {
        return (getThreadsPerInstance() * Nd4j.sizeOfDataType()) + 512;
    }

    /**
     * This method returns desired number of threads per Aggregate instance
     * PLEASE NOTE: this method is especially important for CUDA backend. On CPU backend it might be ignored, depending on Aggregate.
     *
     * @return
     */
    @Override
    public int getThreadsPerInstance() {
        if (vectorLength > 768)
            return 768;

        return vectorLength;
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String name() {
        return "aggregate_hs";
    }

    @Override
    public int maxArguments() {
        return 4;
    }

    @Override
    public int maxShapes() {
        return 0;
    }

    @Override
    public int maxIntArrays() {
        return 0;
    }

    @Override
    public int maxIntArraySize() {
        return 0;
    }

    @Override
    public int maxIndexArguments() {
        return 5;
    }

    @Override
    public int maxRealArguments() {
        return 1;
    }
}
