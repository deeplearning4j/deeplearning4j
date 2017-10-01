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

package org.nd4j.linalg.api.ops.impl.scalar;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;
import org.nd4j.linalg.api.ops.Op;

import java.util.List;

/**
 * Scalar max operation.
 * Returns the max of an element
 * in the ndarray of the specified number.
 *
 * @author Adam Gibson
 */
public class ScalarSet extends BaseScalarOp {
    public ScalarSet() {}

    public ScalarSet(INDArray x, INDArray y, INDArray z, long n, Number num) {
        super(x, y, z, n, num);
    }

    public ScalarSet(INDArray x, Number num) {
        super(x, num);
    }

    public ScalarSet(INDArray x, INDArray y, INDArray z, long n, IComplexNumber num) {
        super(x, y, z, n, num);
    }

    public ScalarSet(SameDiff sameDiff, DifferentialFunction i_v, Number scalar) {
        super(sameDiff, i_v, scalar);
    }

    public ScalarSet(SameDiff sameDiff, DifferentialFunction i_v, Number scalar, boolean inPlace) {
        super(sameDiff, i_v, scalar, inPlace);
    }

    public ScalarSet(SameDiff sameDiff, DifferentialFunction i_v, Number scalar, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, scalar, inPlace, extraArgs);
    }

    public ScalarSet(SameDiff sameDiff, DifferentialFunction i_v, Number scalar, Object[] extraArgs) {
        super(sameDiff, i_v, scalar, extraArgs);
    }

    public ScalarSet(INDArray x, IComplexNumber num) {
        super(x, num);
    }

    @Override
    public int opNum() {
        return 13;
    }

    @Override
    public String name() {
        return "set_scalar";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return complexNumber;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return complexNumber;

    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return complexNumber;

    }

    @Override
    public float op(float origin, float other) {
        return num.floatValue();
    }

    @Override
    public double op(double origin, double other) {
        return num.doubleValue();
    }

    @Override
    public double op(double origin) {
        return num.doubleValue();

    }

    @Override
    public float op(float origin) {
        return num.floatValue();

    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return complexNumber;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        if (num != null)
            return new ScalarSet(x.vectorAlongDimension(index, dimension), num);
        else
            return new ScalarSet(x.vectorAlongDimension(index, dimension), complexNumber);
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        if (num != null)
            return new ScalarSet(x.tensorAlongDimension(index, dimension), num);
        else
            return new ScalarSet(x.tensorAlongDimension(index, dimension), complexNumber);
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        if (num != null) {
            this.extraArgs = new Object[] {num};
        } else {
            this.extraArgs = new Object[] {complexNumber};
        }

    }


    /**
     * Get the value of this function
     *
     * @return
     */
    @Override
    public ArrayField doGetValue() {
        if(scalarValue == null) {
            scalarValue = (Number) extraArgs[0];
        }

        return arg().getValue(true).set(scalarValue.doubleValue());
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v1) {
        throw new UnsupportedOperationException();
    }

}
