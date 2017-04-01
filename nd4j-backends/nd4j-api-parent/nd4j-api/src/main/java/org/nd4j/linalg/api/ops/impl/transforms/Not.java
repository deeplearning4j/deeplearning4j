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

package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.NonNull;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;

/**
 * Boolean AND pairwise transform
 *
 * @author raver119@gmail.com
 */
public class Not extends BaseTransformOp {

    protected double comparable;

    public Not() {}

    public Not(@NonNull INDArray x) {
        this(x, 0.0);
    }

    public Not(@NonNull INDArray x, Number comparable) {
        this(x, x, comparable, x.lengthLong());
    }

    public Not(@NonNull INDArray x, INDArray z, Number comparable) {
        this(x, z, comparable, x.lengthLong());
    }

    public Not(@NonNull INDArray x, INDArray z) {
        this(x, z, z.lengthLong());
    }

    public Not(@NonNull INDArray x, INDArray z, long n) {
        this(x, z, 0.0, n);
    }

    public Not(@NonNull INDArray x, INDArray z, Number comparable, long n) {
        super(x, null, z, n);
        this.comparable = comparable.doubleValue();
        this.extraArgs = new Object[] {this.comparable};
    }


    @Override
    public int opNum() {
        return 59;
    }

    @Override
    public String name() {
        return "boolean_not";
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
        return new Not(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);
        return new Not(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }
}
