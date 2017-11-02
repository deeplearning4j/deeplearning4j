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

package org.nd4j.linalg.api.ops.impl.transforms.arithmetic;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;

import java.util.ArrayList;
import java.util.List;

/**
 * Truncated division operation
 *
 * @author Adam Gibson
 */
public class FloorDivOp extends BaseTransformOp {
    public FloorDivOp(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public FloorDivOp(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public FloorDivOp() {}

    public FloorDivOp(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public FloorDivOp(INDArray x) {
        super(x);
    }

    public FloorDivOp(INDArray x, INDArray z) {
        super(x, z);
    }

    public FloorDivOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public FloorDivOp(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    @Override
    public int opNum() {
        return 20;
    }

    @Override
    public String name() {
        return "floordiv";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin.div(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return origin.div(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return origin.div(other);
    }

    @Override
    public float op(float origin, float other) {
        return origin / other;
    }

    @Override
    public double op(double origin, double other) {
        return origin / other;
    }

    @Override
    public double op(double origin) {
        return origin;
    }

    @Override
    public float op(float origin) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return origin;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new FloorDivOp(xAlongDimension, y.vectorAlongDimension(index, dimension),
                            z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new FloorDivOp(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new FloorDivOp(xAlongDimension, y.tensorAlongDimension(index, dimension),
                            z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new FloorDivOp(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }


    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        if (y == null)
            throw new IllegalArgumentException("No components to divide");
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction gradWrtX = f().div(i_v.get(0),rarg());
        DifferentialFunction gradWrtY = f().mul(f().neg(gradWrtX),f().div(larg(),rarg()));
        List<DifferentialFunction> ret = new ArrayList<>(2);
        ret.add(gradWrtX);
        ret.add(gradWrtY);
        return ret;
    }


}
