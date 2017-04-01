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
 * Element-wise Compare-and-set implementation as Op
 *
 * Please check javadoc to specific constructors, for detail information.
 *
 * @author raver119@gmail.com
 */
public class CompareAndSet extends BaseTransformOp {

    private double compare;
    private double set;
    private double eps;
    private int mode;


    public CompareAndSet() {

    }

    public CompareAndSet(INDArray x, double compare, double set, double eps) {
        this(x, compare, set, eps, null);
    }

    public CompareAndSet(INDArray x, double compare, double set, double eps, Condition condition) {
        super(x);
        this.compare = compare;
        this.set = set;
        this.eps = eps;
        if (condition == null)
            this.mode = 0;
        else
            this.mode = condition.condtionNum();

        init(x, null, x, x.length());
    }


    /**
     * With this constructor, op will check each X element against given Condition, and if condition met, element will be replaced with Set value
     *
     *
     * Pseudocode:
     * z[i] = condition(x[i]) ? set : x[i];
     *
     * PLEASE NOTE: X will be modified inplace.
     *
     * @param x
     * @param set
     * @param condition
     */
    public CompareAndSet(INDArray x, double set, Condition condition) {
        this(x, x, set, condition);
    }


    /**
     * With this constructor, op will check each X element against given Condition, and if condition met, element will be replaced with Set value
     *
     * Pseudocode:
     * z[i] = condition(x[i]) ? set : x[i];
     *
     * @param x
     * @param set
     * @param condition
     */
    public CompareAndSet(INDArray x, INDArray z, double set, Condition condition) {
        super(x, null, z, x.lengthLong());
        this.compare = condition.getValue();
        this.set = set;
        this.mode = condition.condtionNum();
        init(x, null, z, x.lengthLong());
    }

    /**
     * With this constructor, op will check each Y element against given Condition, and if condition met, element Z will be set to Y value, and X otherwise
     *
     * PLEASE NOTE: X will be modified inplace.
     *
     * Pseudocode:
     * z[i] = condition(y[i]) ? y[i] : x[i];
     *
     * @param x
     * @param y
     * @param condition
     */
    public CompareAndSet(INDArray x, INDArray y, Condition condition) {
        this(x, y, x, condition);
    }


    /**
     * With this constructor, op will check each Y element against given Condition, and if condition met, element Z will be set to Y value, and X otherwise
     *
     * Pseudocode:
     * z[i] = condition(y[i]) ? y[i] : x[i];
     *
     * @param x
     * @param y
     * @param z
     * @param condition
     */
    public CompareAndSet(INDArray x, INDArray y, INDArray z, Condition condition) {
        super(x, y, z, x.lengthLong());
        this.compare = condition.getValue();
        this.set = 0;
        this.mode = condition.condtionNum();
        init(x, y, z, x.lengthLong());
    }

    /**
     * This constructor is shortcut to epsEquals.
     *
     * @param x
     * @param z
     * @param compare
     * @param set
     * @param eps
     */
    public CompareAndSet(INDArray x, INDArray z, double compare, double set, double eps) {
        super(x, z);
        this.compare = compare;
        this.set = set;
        this.eps = eps;
        this.mode = 0;
        init(x, null, z, x.length());
    }

    /**
     * This constructor is shortcut to epsEquals.
     *
     * @param x
     * @param z
     * @param compare
     * @param set
     * @param eps
     */
    public CompareAndSet(INDArray x, INDArray z, double compare, double set, double eps, long n) {
        super(x, z, n);
        this.compare = compare;
        this.set = set;
        this.eps = eps;
        this.mode = 0;
        init(x, null, x, n);
    }

    @Override
    public int opNum() {
        return 45;
    }

    @Override
    public String name() {
        return "cas";
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

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new CompareAndSet(xAlongDimension, z.vectorAlongDimension(index, dimension), compare, set, eps,
                            xAlongDimension.length());
        else
            return new CompareAndSet(xAlongDimension, z.vectorAlongDimension(index, dimension), compare, set, eps,
                            xAlongDimension.length());
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new CompareAndSet(xAlongDimension, z.tensorAlongDimension(index, dimension), compare, set, eps,
                            xAlongDimension.length());
        else
            return new CompareAndSet(xAlongDimension, z.tensorAlongDimension(index, dimension), compare, set, eps,
                            xAlongDimension.length());

    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[] {compare, set, eps, (double) mode};
    }
}

