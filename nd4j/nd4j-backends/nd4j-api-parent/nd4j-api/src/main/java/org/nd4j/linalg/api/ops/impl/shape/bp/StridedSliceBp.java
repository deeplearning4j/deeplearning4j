/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.shape.bp;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

    private void addArguments(){
        addIArgument(beginMask);
        addIArgument(ellipsisMask);
        addIArgument(endMask);
        addIArgument(newAxisMask);
        addIArgument(shrinkAxisMask);
        addIArgument(begin);
        addIArgument(end);
        addIArgument(strides);
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

}
