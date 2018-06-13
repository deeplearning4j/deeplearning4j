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

package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Calculate the mean of the vector
 *
 * @author Adam Gibson
 */
public class Mean extends Sum {
    public Mean(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public Mean(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public Mean() {
    }

    public Mean(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Mean(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public Mean(INDArray x) {
        super(x);
    }

    public Mean(INDArray x, INDArray y) {
        super(x, y);
    }

    public Mean(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "mean";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        //If out = mean(in), then dL/dIn = 1/N * dL/dOut  (broadcast to appropriate shape)
        //Note that N differs for "along dimension" vs. "whole array" reduce cases
        return Collections.singletonList(f().meanBp(arg(), i_v1.get(0), false, dimensions));
    }

    @Override
    public String onnxName() {
        return "ReduceMean";
    }

    @Override
    public String tensorflowName() {
        return "Mean";
    }
}
