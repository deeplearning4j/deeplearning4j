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

package org.nd4j.linalg.api.ops.aggregates;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public abstract class   BaseAggregate implements Aggregate {
    protected List<INDArray> arguments = new ArrayList<>();
    protected List<DataBuffer> shapes = new ArrayList<>();
    protected List<int[]> intArrayArguments = new ArrayList<>();
    protected List<Integer> indexingArguments = new ArrayList<>();
    protected List<Number> realArguments = new ArrayList<>();

    protected Number finalResult = 0.0;

    public List<INDArray> getArguments() {
        return arguments;
    }

    @Override
    public Number getFinalResult() {
        return finalResult;
    }

    @Override
    public void setFinalResult(Number result) {
        this.finalResult = result;
    }

    @Override
    public List<DataBuffer> getShapes() {
        return shapes;
    }

    @Override
    public List<Integer> getIndexingArguments() {
        return indexingArguments;
    }

    @Override
    public List<Number> getRealArguments() {
        return realArguments;
    }

    @Override
    public List<int[]> getIntArrayArguments() {
        return intArrayArguments;
    }

    @Override
    public long getRequiredBatchMemorySize() {
        long result = maxIntArrays() * maxIntArraySize() * 4;
        result += maxArguments() * 8; // pointers
        result += maxShapes() * 8; // pointers
        result += maxIndexArguments() * 4;
        result += maxRealArguments() * (Nd4j.dataType() == DataBuffer.Type.DOUBLE ? 8
                        : Nd4j.dataType() == DataBuffer.Type.FLOAT ? 4 : 2);
        result += 5 * 4; // numArgs

        return result * Batch.getBatchLimit();
    }
}
