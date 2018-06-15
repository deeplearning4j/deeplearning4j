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

import lombok.NonNull;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseIndexAccumulation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.util.Collections;
import java.util.List;

/**
 * Calculate the index
 * of max value over a vector
 *
 * @author raver119@gmail.com
 */
public class FirstIndex extends BaseIndexAccumulation {
    protected Condition condition;
    protected double compare;
    protected double eps;
    protected int mode;

    public FirstIndex(SameDiff sameDiff, SDVariable i_v, Condition condition, int... dimensions) {
        super(sameDiff, i_v, dimensions);
        this.condition = condition;
        this.compare = condition.getValue();
        this.mode = condition.condtionNum();
        this.eps = eps;
        this.extraArgs = new Object[] {compare, eps, (double) mode};
    }

    public FirstIndex() {}


    public FirstIndex(INDArray x, @NonNull Condition condition) {
        this(x, condition, Nd4j.EPS_THRESHOLD);
    }

    public FirstIndex(INDArray x, @NonNull Condition condition, double eps) {
        super(x);
        this.condition = condition;
        this.compare = condition.getValue();
        this.mode = condition.condtionNum();
        this.eps = eps;
        this.extraArgs = new Object[] {compare, eps, (double) mode};
    }


    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String opName() {
        return "first_index";
    }

    @Override
    public float zeroFloat() {
        return 0.0f;
    }

    @Override
    public float zeroHalf() {
        return zeroFloat();
    }

    @Override
    public double zeroDouble() {
        return 0.0;
    }

    @Override
    public IComplexNumber zeroComplex() {
        return Nd4j.createComplexNumber(-Double.MAX_VALUE, 0);
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

}
