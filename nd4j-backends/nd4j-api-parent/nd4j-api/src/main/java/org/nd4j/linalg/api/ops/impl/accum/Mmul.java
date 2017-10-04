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

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.shape.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * Matrix multiplication/dot product
 *
 * @author Adam Gibson
 */
public class Mmul extends TensorMmul {

    private MMulTranspose mMulTranspose;

    /**
     *
     * @param sameDiff
     * @param i_v1
     * @param i_v2
     * @param mMulTranspose
     */
    public Mmul(SameDiff sameDiff,
                DifferentialFunction i_v1,
                DifferentialFunction i_v2,
                MMulTranspose mMulTranspose) {
        super(sameDiff,
                i_v1,
                i_v2, new int[][] {
                        {1},{0}
                },mMulTranspose);

        this.mMulTranspose = mMulTranspose;
    }


    /**
     *
     * @param sameDiff
     * @param i_v1
     * @param i_v2
     */
    public Mmul(SameDiff sameDiff,
                DifferentialFunction i_v1,
                DifferentialFunction i_v2) {
        this(sameDiff,i_v1,i_v2,MMulTranspose.allFalse());
    }

    public Mmul(INDArray x, INDArray y, int[][] axes, MMulTranspose mMulTranspose) {
        super(x, y, axes);
        this.mMulTranspose = mMulTranspose;
    }

    public Mmul(INDArray x, INDArray y, INDArray z, MMulTranspose mMulTranspose) {
        super(x, y, z,  new int[][] {
                {1},{0}
        });
        this.mMulTranspose = mMulTranspose;
    }

    public Mmul() {}









    @Override
    public int opNum() {
        return 3;
    }

    @Override
    public String name() {
        return "mmul";
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        throw new UnsupportedOperationException();


    }

    @Override
    public long n() {
        return 0;
    }

    @Override
    public boolean isPassThrough() {
        return true;
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        throw new UnsupportedOperationException();

    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        numProcessed++;
        return origin.mul(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        numProcessed++;
        return origin.mul(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        numProcessed++;
        return origin.mul(other);
    }

    @Override
    public float op(float origin, float other) {
        numProcessed++;
        return origin * other;
    }

    @Override
    public double op(double origin, double other) {
        numProcessed++;
        return origin * other;
    }

    @Override
    public double op(double origin) {
        numProcessed++;
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
    public double update(double accum, double x) {
        return accum + x;
    }

    @Override
    public double update(double accum, double x, double y) {
        return accum + x * y;
    }

    @Override
    public float update(float accum, float x) {
        return accum + x;
    }

    @Override
    public float update(float accum, float x, float y) {
        return accum + x * y;
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, double x) {
        return accum.add(x);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, double x, double y) {
        return accum.add(x * y);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x) {
        return accum.add(x);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, IComplexNumber y) {
        return accum.add(x.mul(y));
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        return accum.add(x.mul(y));
    }

    @Override
    public double combineSubResults(double first, double second) {
        return first + second;
    }

    @Override
    public float combineSubResults(float first, float second) {
        return first + second;
    }

    @Override
    public IComplexNumber combineSubResults(IComplexNumber first, IComplexNumber second) {
        return first.add(second);
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public void exec() {
        if(this.z != null)
            x.mmul(y,z,mMulTranspose);
        else
            this.z = x.mmul(y,mMulTranspose);
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        DifferentialFunction setup = sameDiff.setupFunction(i_v1.get(0));
        DifferentialFunction gradWrtX = sameDiff.setupFunction(f().reshape(f().mmul(setup,rarg(),
                MMulTranspose.builder()
                        .transposeB(!mMulTranspose.isTransposeB())
                        .transposeResult(mMulTranspose.isTransposeA())
                        .build()),larg().getResultShape()));

        DifferentialFunction gradWrtY = sameDiff.setupFunction(f().reshape(f().mmul(larg(),setup,
                MMulTranspose.builder()
                        .transposeA(!mMulTranspose.isTransposeA())
                        .transposeResult(mMulTranspose.isTransposeB())
                        .build()),rarg().getResultShape()));

        ret.add(gradWrtX);
        ret.add(gradWrtY);
        validateFunctionReference(larg());
        validateFunctionReference(rarg());
        return ret;
    }

    @Override
    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction i_v1,
                            DifferentialFunction i_v2,
                            String opName) {
           //skip empty dimensions
        addEdges(sameDiff,i_v1,i_v2,opName,
                Type.REDUCE3,
                Shape.getMatrixMultiplyShape(
                        i_v1.getResultShape(),
                        i_v2.getResultShape()));
    }



    /**
     * Get the value of this function
     *
     * @return
     */
    @Override
    public ArrayField doGetValue() {
        return a().mmul(larg(),rarg(),mMulTranspose);
    }

}

