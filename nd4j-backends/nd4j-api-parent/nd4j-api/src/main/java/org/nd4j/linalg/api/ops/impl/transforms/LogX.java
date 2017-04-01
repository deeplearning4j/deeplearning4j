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

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Log on arbitrary base op
 *
 * @author raver119@gmail.com
 */
public class LogX extends BaseTransformOp {
    private double base;

    public LogX() {}

    public LogX(INDArray x, INDArray z, double base) {
        super(x, z);
        this.base = base;
        this.extraArgs = new Object[] {base};
    }

    public LogX(INDArray x, INDArray z, double base, long n) {
        super(x, z, n);
        this.base = base;
        this.extraArgs = new Object[] {base};
    }

    public LogX(INDArray x, INDArray y, INDArray z, double base, long n) {
        super(x, y, z, n);
        this.base = base;
        this.extraArgs = new Object[] {base};
    }

    public LogX(INDArray x, double base) {
        super(x);
        this.base = base;
        this.extraArgs = new Object[] {base};
    }

    @Override
    public int opNum() {
        return 55;
    }

    @Override
    public String name() {
        return "log_x";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return ComplexUtil.log(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return ComplexUtil.log(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return ComplexUtil.log(origin);
    }

    @Override
    public float op(float origin, float other) {
        return (float) FastMath.log(origin) / (float) FastMath.log((float) base);
    }

    @Override
    public double op(double origin, double other) {
        return FastMath.log(origin) / FastMath.log(base);
    }

    @Override
    public double op(double origin) {
        return FastMath.log(origin) / FastMath.log(base);
    }

    @Override
    public float op(float origin) {
        return (float) FastMath.log(origin) / (float) FastMath.log(base);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return ComplexUtil.log(origin);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new LogX(xAlongDimension, y.vectorAlongDimension(index, dimension),
                            z.vectorAlongDimension(index, dimension), base, xAlongDimension.length());
        else
            return new LogX(xAlongDimension, z.vectorAlongDimension(index, dimension), base, x.lengthLong());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new LogX(xAlongDimension, y.tensorAlongDimension(index, dimension),
                            z.tensorAlongDimension(index, dimension), base, xAlongDimension.length());
        else
            return new LogX(xAlongDimension, z.tensorAlongDimension(index, dimension), base, x.lengthLong());

    }
}
