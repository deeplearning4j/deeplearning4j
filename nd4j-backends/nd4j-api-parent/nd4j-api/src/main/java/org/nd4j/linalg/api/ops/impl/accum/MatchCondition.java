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

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;

/**
 * Absolute sum the components
 *
 * @author raver119@gmail.com
 */
public class MatchCondition extends BaseAccumulation {

    double compare;
    double eps;
    int mode;

    public MatchCondition() {}


    public MatchCondition(INDArray x, Condition condition) {
        this(x, Nd4j.EPS_THRESHOLD, condition);
    }

    public MatchCondition(INDArray x, double eps, Condition condition) {
        super(x);
        this.compare = condition.getValue();
        this.mode = condition.condtionNum();
        this.eps = eps;

        this.extraArgs = new Object[] {compare, eps, (double) mode};
    }

    @Override
    public int opNum() {
        return 12;
    }

    @Override
    public String name() {
        return "match_condition";
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        return null;
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        return null;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return null;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return null;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return null;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return null;
    }

    @Override
    public double update(double accum, double x) {
        return 0;
    }

    @Override
    public double update(double accum, double x, double y) {
        return 0;
    }

    @Override
    public float update(float accum, float x) {
        return 0;
    }

    @Override
    public float update(float accum, float x, float y) {
        return 0;
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, double x) {
        return null;
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, double x, double y) {
        return null;
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x) {
        return null;
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, IComplexNumber y) {
        return null;
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        return null;
    }
}
