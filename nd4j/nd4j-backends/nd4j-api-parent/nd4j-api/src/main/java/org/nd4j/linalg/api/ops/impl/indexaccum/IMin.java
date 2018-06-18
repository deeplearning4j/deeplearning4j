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

package org.nd4j.linalg.api.ops.impl.indexaccum;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseIndexAccumulation;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Calculate the index of min value over a vector
 *
 * @author Alex Black
 */
public class IMin extends BaseIndexAccumulation {
    public IMin(SameDiff sameDiff, SDVariable i_v, boolean keepDims, int[] dimensions) {
        super(sameDiff, i_v, keepDims, dimensions);
    }

    public IMin() {
    }

    public IMin(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public IMin(INDArray x) {
        super(x);
    }

    public IMin(INDArray x, INDArray y) {
        super(x, y);
    }


    @Override
    public int opNum() {
        return 1;
    }

    @Override
    public String opName() {
        return "imin";
    }


    @Override
    public float zeroFloat() {
        return Float.MAX_VALUE;
    }

    @Override
    public double zeroDouble() {
        return Double.MAX_VALUE;
    }

    @Override
    public float zeroHalf() {
        return 65503.0f;
    }

    @Override
    public IComplexNumber zeroComplex() {
        return Nd4j.createComplexNumber(Double.MAX_VALUE, 0);
    }


    @Override
    public String onnxName() {
        return "ArgMin";
    }

    @Override
    public String tensorflowName() {
        return "argmin";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //Not differentiable, but (assuming no ties) output does not change for a given infinitesimal change in the input

        return Collections.singletonList(f().zerosLike(arg()));
    }
}
