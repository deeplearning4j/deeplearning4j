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

package org.nd4j.linalg.api.ops.impl.transforms.comparison;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.indexing.conditions.Condition;

/**
 * Element-wise Compare-and-Replace implementation as Op
 * Basically this op does the same as Compare-and-Set, but op.X is checked against Condition instead
 *
 * @author raver119@gmail.com
 */
public class CompareAndReplace extends BaseTransformOp {

    private double compare;
    private double set;
    private double eps;
    private int mode;


    public CompareAndReplace() {

    }


    /**
     * With this constructor, op will check each X element against given Condition, and if condition met, element Z will be set to Y value, and X otherwise
     *
     * PLEASE NOTE: X will be modified inplace.
     *
     * Pseudocode:
     * z[i] = condition(x[i]) ? y[i] : x[i];
     *
     * @param x
     * @param y
     * @param condition
     */
    public CompareAndReplace(INDArray x, INDArray y, Condition condition) {
        this(x, y, x, condition);
    }

    /**
     * With this constructor, op will check each X element against given Condition, and if condition met, element Z will be set to Y value, and X otherwise
     *
     * Pseudocode:
     * z[i] = condition(x[i]) ? y[i] : x[i];
     *
     * @param x
     * @param y
     * @param z
     * @param condition
     */
    public CompareAndReplace(INDArray x, INDArray y, INDArray z, Condition condition) {
        super(x, y, z, x.lengthLong());
        this.compare = condition.getValue();
        this.set = 0;
        this.mode = condition.condtionNum();
        init(x, y, z, x.lengthLong());
    }



    @Override
    public int opNum() {
        return 46;
    }

    @Override
    public String name() {
        return "car";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return null;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return null;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return null;
    }

    @Override
    public float op(float origin, float other) {
        return 0;
    }

    @Override
    public double op(double origin, double other) {
        return 0;
    }

    @Override
    public double op(double origin) {
        return 0;
    }

    @Override
    public float op(float origin) {
        return 0;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return null;

    }

    /**
     * A copy of this operation for a particular dimension of the input
     *
     * @param index     the index of the op to iterate over
     * @param dimension the dimension to ge the input for
     * @return the operation for that dimension
     */
    @Override
    public Op opForDimension(int index, int dimension) {
        return null;
    }

    /**
     * A copy of this operation for a particular dimension of the input
     *
     * @param index     the index of the op to iterate over
     * @param dimension the dimension to ge the input for
     * @return the operation for that dimension
     */
    @Override
    public Op opForDimension(int index, int... dimension) {
        return null;
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[] {compare, set, eps, (double) mode};
    }
}

