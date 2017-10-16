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

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.impl.OneInitScheme;

import java.util.Arrays;
import java.util.List;

/**
 * Ones (represents a constant)
 *
 * @author Adam Gibson
 */
public class Ones extends Constant {

    public Ones() {
    }


    public Ones(SameDiff sameDiff, int[] shape,int vertexId) {
        super(sameDiff,NDArrayInformation.newInfo(shape,new OneInitScheme('f')), shape,vertexId);
    }

    public Ones(SameDiff sameDiff, NDArrayInformation i_v, int[] shape, boolean inPlace,int vertexId) {
        super(sameDiff, i_v, shape, inPlace,vertexId);
    }

    public Ones(SameDiff sameDiff, NDArrayInformation i_v, int[] shape,int vertexId) {
        super(sameDiff, i_v, shape,vertexId);
    }

    public Ones(INDArray x, INDArray z, int[] shape) {
        super(x, z, shape);
    }

    public Ones(int[] shape) {
        super(shape);
    }

    public Ones(INDArray x, INDArray z, long n, int[] shape) {
        super(x, z, n, shape);
    }

    public Ones(INDArray x, INDArray y, INDArray z, long n, int[] shape) {
        super(x, y, z, n, shape);
    }

    public Ones(INDArray x, int[] shape) {
        super(x, shape);
    }

    @Override
    public int opNum() {
        return 26;
    }

    @Override
    public String name() {
        return "ones";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return Nd4j.createComplexNumber(1, 1);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return Nd4j.createComplexNumber(1, 1);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return Nd4j.createComplexNumber(1, 1);
    }

    @Override
    public float op(float origin, float other) {
        return 1;
    }

    @Override
    public double op(double origin, double other) {
        return 1;
    }

    @Override
    public double op(double origin) {
        return 1;
    }

    @Override
    public float op(float origin) {
        return 1;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return Nd4j.createComplexNumber(1, 1);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
       throw new UnsupportedOperationException();

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        throw new UnsupportedOperationException();

    }

    @Override
    public INDArray getZ() {
        return z();
    }

    @Override
    public INDArray z() {
        return Nd4j.ones(shape);
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return Arrays.asList((DifferentialFunction) f().zero(f1.get(0).getResultShape()));
    }
}
