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
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;

import java.util.Arrays;
import java.util.List;

/**
 * Operation for fast INDArrays equality checks
 *
 * @author raver119@gmail.com
 */
public class EqualsWithEps extends BaseAccumulation {
    private double eps;

    public EqualsWithEps(SameDiff sameDiff, SDVariable i_v, int[] dimensions, double eps) {
        super(sameDiff, i_v, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, double eps) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps() {}

    public EqualsWithEps(INDArray x, INDArray y, INDArray z, long n, double eps) {
        super(x, y, z, n);
        this.extraArgs = new Object[] {eps};
    }

    public EqualsWithEps(INDArray x, INDArray y, long n, double eps) {
        super(x, y, n);
        this.extraArgs = new Object[] {eps};
    }

    public EqualsWithEps(INDArray x, INDArray y, double eps) {
        super(x, y);
        this.extraArgs = new Object[] {eps};
    }

    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String opName() {
        return "equals_with_eps";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Arrays.asList(outputVariables()[0]);
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

    @Override
    public Type getOpType() {
        return Type.REDUCE3;
    }
}
