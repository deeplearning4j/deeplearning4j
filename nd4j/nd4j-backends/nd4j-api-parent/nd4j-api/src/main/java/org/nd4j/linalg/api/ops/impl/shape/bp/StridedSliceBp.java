/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.shape.bp;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.Collections;
import java.util.List;

/**
 * Strided Slice backprop function
 *
 * @author Alex Black
 */
@Slf4j
public class StridedSliceBp extends DynamicCustomOp {
    private long[] begin;
    private long[] end;
    private long[] strides;
    private int beginMask;
    private int endMask;
    private int ellipsisMask;
    private int newAxisMask;
    private int shrinkAxisMask;

    public StridedSliceBp() {}

    public StridedSliceBp(SameDiff sameDiff, @NonNull SDVariable in, @NonNull SDVariable grad, @NonNull long[] begin, @NonNull long[] end, @NonNull long[] strides,
                          int beginMask, int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask){
        super(null, sameDiff, new SDVariable[]{in, grad});
        this.begin = begin;
        this.end = end;
        this.strides = strides;
        this.beginMask = beginMask;
        this.endMask = endMask;
        this.ellipsisMask = ellipsisMask;
        this.newAxisMask = newAxisMask;
        this.shrinkAxisMask = shrinkAxisMask;

        //https://github.com/deeplearning4j/libnd4j/blob/master/include/ops/declarable/generic/parity_ops/strided_slice.cpp#L279
        addArguments();
    }

    public StridedSliceBp(SameDiff sameDiff, @NonNull SDVariable in, @NonNull SDVariable grad, @NonNull SDVariable begin, @NonNull SDVariable end,
                          @NonNull SDVariable strides, int beginMask, int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask){
        super(null, sameDiff, new SDVariable[]{in, grad, begin, end, strides});
        this.beginMask = beginMask;
        this.endMask = endMask;
        this.ellipsisMask = ellipsisMask;
        this.newAxisMask = newAxisMask;
        this.shrinkAxisMask = shrinkAxisMask;
        addArguments();
    }

    private void addArguments(){
        addIArgument(beginMask);
        addIArgument(ellipsisMask);
        addIArgument(endMask);
        addIArgument(newAxisMask);
        addIArgument(shrinkAxisMask);
        if(begin != null) { //May be null for SDVariable inputs of these args
            addIArgument(begin);
            addIArgument(end);
            addIArgument(strides);
        }
    }


    @Override
    public String opName() {
        return "strided_slice_bp";
    }


    @Override
    public void assertValidForExecution() {
        if(numInputArguments() != 2 && numInputArguments() != 4) {
            throw new ND4JIllegalStateException("Num input arguments must be 2 or 4.");
        }

        if(numIArguments() < 5) {
            throw new ND4JIllegalStateException("Number of integer arguments must >= 5");
        }
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Differentation not supported for backprop function: " + getClass().getSimpleName());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes.size() == 2 || dataTypes.size() == 5, "Expected list with exactly 2 or 5 datatypes for %s, got %s", getClass(), dataTypes);
        //Output type is same as (original) input type
        return Collections.singletonList(arg().dataType());
    }
}
