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

package org.nd4j.linalg.jcublas.ops.executioner;

import lombok.NonNull;
import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOpContext;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.nativeblas.Nd4jCuda;

/**
 * CUDA wrapper for op Context
 * @author raver119@gmail.com
 */
public class CudaOpContext extends BaseOpContext implements OpContext {
    // we might want to have configurable
    private Nd4jCuda.Context context = new Nd4jCuda.Context(1);

    @Override
    public void setIArguments(long... arguments) {
        super.setIArguments(arguments);
        context.setIArguments(arguments, arguments.length);
    }

    @Override
    public void setBArguments(boolean... arguments) {
        super.setBArguments(arguments);
        context.setBArguments(arguments, arguments.length);
    }

    @Override
    public void setTArguments(double... arguments) {
        super.setTArguments(arguments);
        context.setTArguments(arguments, arguments.length);
    }

    @Override
    public void setRngStates(long rootState, long nodeState) {
        context.randomGenerator().setStates(rootState, nodeState);
    }

    @Override
    public Pair<Long, Long> getRngStates() {
        return Pair.makePair(context.randomGenerator().rootState(), context.randomGenerator().nodeState());
    }

    @Override
    public void setInputArray(int index, @NonNull INDArray array) {
        context.setInputArray(index, array.data().addressPointer(), array.shapeInfoDataBuffer().addressPointer(), null, null);

        super.setInputArray(index, array);
    }

    @Override
    public void setOutputArray(int index, @NonNull INDArray array) {
        context.setOutputArray(index, array.data().addressPointer(), array.shapeInfoDataBuffer().addressPointer(), null, null);

        super.setOutputArray(index, array);
    }

    @Override
    public Pointer contextPointer() {
        for (val v:fastpath_in.values())
            AtomicAllocator.getInstance().synchronizeHostData(v);

        for (val v:fastpath_out.values()) {
            AtomicAllocator.getInstance().synchronizeHostData(v);
            AtomicAllocator.getInstance().getAllocationPoint(v).tickHostWrite();
        }

        return context;
    }
}
