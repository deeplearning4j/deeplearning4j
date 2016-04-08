/*
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

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**If x is input: output is x*(1-x)
 * @author Alex Black
 */
public class TimesOneMinus extends BaseTransformOp {

    public TimesOneMinus() {
    }

    public TimesOneMinus(INDArray x, INDArray z) {
        super(x, z);
    }

    public TimesOneMinus(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public TimesOneMinus(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public TimesOneMinus(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 24;
    }

    @Override
    public String name() {
        return "timesoneminus";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return Nd4j.createComplexNumber(1, 1).subi(origin).muli(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return Nd4j.createComplexNumber(1, 1).subi(origin).muli(origin);

    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return Nd4j.createComplexNumber(1, 1).subi(origin).muli(origin);

    }

    @Override
    public float op(float origin, float other) {
        return origin * (1 - origin);
    }

    @Override
    public double op(double origin, double other) {
        return origin * (1 - origin);
    }

    @Override
    public double op(double origin) {
        return origin * (1 - origin);
    }

    @Override
    public float op(float origin) {
        return origin * (1 - origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return Nd4j.createComplexNumber(1, 1).subi(origin).muli(origin);

    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new TimesOneMinus(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new TimesOneMinus(xAlongDimension, z.vectorAlongDimension(index, dimension), x.lengthLong());
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new TimesOneMinus(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new TimesOneMinus(xAlongDimension, z.tensorAlongDimension(index, dimension), x.lengthLong());
    }
}
