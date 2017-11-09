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

import com.google.common.primitives.Ints;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.nd4j.linalg.util.ArrayUtil.*;

/**
 * TensorMmul
 * @author Adam Gibson
 */
@NoArgsConstructor
public class TensorMmul extends BaseAccumulation {
    private int[][] axes;
    protected boolean addedEdges;
    protected MMulTranspose mMulTranspose;

    public TensorMmul(SameDiff sameDiff,
                      DifferentialFunction i_v1,
                      DifferentialFunction i_v2,
                      int[][] dimensions) {
        this(sameDiff,i_v1,i_v2,dimensions,MMulTranspose.allFalse());
    }

    public TensorMmul(SameDiff sameDiff,
                      DifferentialFunction i_v1,
                      DifferentialFunction i_v2,
                      int[][] dimensions,
                      MMulTranspose mMulTranspose) {
        super(sameDiff);
        this.sameDiff = sameDiff;
        this.mMulTranspose = mMulTranspose;
        this.axes = dimensions;
        this.extraArgs = new Object[] {axes,mMulTranspose};

        this.args = new DifferentialFunction[] {i_v1,i_v2};
        f().validateFunctionReference(i_v1);
        f().validateFunctionReference(i_v2);
        this.shape = calculateOutputShape().get(0);
        addAsNewVertexId();
        if(!addedEdges) {
            f().addFunctionEdges(this);
            addedEdges = true;
        }
    }

    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>(1);
        int[] aShape = mMulTranspose.isTransposeA() ? ArrayUtil.reverseCopy(larg().getResultShape()) : larg().getResultShape();
        int[] bShape = mMulTranspose.isTransposeB() ? ArrayUtil.reverseCopy(rarg().getResultShape()) : rarg().getResultShape();

        ret.add(  this instanceof Mmul ? Shape.getMatrixMultiplyShape(aShape,bShape)
                : getTensorMmulShape(aShape,bShape, axes));
        return ret;
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        int[] bAxes = range(0, rarg().getResultShape().length);
        int[] aAxes = range(0, larg().getResultShape().length);
        int aRank = larg().getResultShape().length;
        int bRank = rarg().getResultShape().length;
        int[][] sumAxes = new int[][]{
                mod(axes[0], aRank), mod(axes[1], bRank)
        };
        int[][] deletedAxes = new int[][]{
                removeIndex(aAxes, sumAxes[0]),
                removeIndex(bAxes, sumAxes[1])};
        int[] gAxes = range(0, i_v1.get(0).getResultShape().length);
        int[][] firstAxes = new int[][]{
                Arrays.copyOfRange(gAxes, deletedAxes[0].length, gAxes.length),
                deletedAxes[1]
        };

        int[][] secondAxes = new int[][]{
                deletedAxes[0],
                Arrays.copyOfRange(gAxes, 0, deletedAxes[0].length)
        };


        //tensor matrix multiply gradient wrt second variable
        int[] firstPerm = argsort(combine(deletedAxes[0],keep(argsort(sumAxes[1]),sumAxes[0])));
        DifferentialFunction firstResult = doTensorMmul(i_v1.get(0), rarg(), firstAxes);
        DifferentialFunction permuted = f().permute(firstResult,firstPerm);
        ret.add(permuted);

        //tensor matrix multiply gradient wrt first variable
        int[] secondPerm = argsort(combine(keep(argsort(sumAxes[0]),sumAxes[1]),deletedAxes[1]));
        DifferentialFunction secondResult = doTensorMmul(i_v1.get(0), larg(), secondAxes);
        DifferentialFunction secondPermuted = f().permute(secondResult,secondPerm);
        ret.add(secondPermuted);
        return ret;
    }



    private DifferentialFunction doTensorMmul(DifferentialFunction a,
                                              DifferentialFunction b,
                                              int[][] axes) {

        int validationLength = Math.min(axes[0].length, axes[1].length);
        for (int i = 0; i < validationLength; i++) {
            if (a.getResultShape()[axes[0][i]] != b.getResultShape()[axes[1][i]])
                throw new IllegalArgumentException("Size of the given axes at each dimension must be the same size.");
            if (axes[0][i] < 0)
                axes[0][i] += a.getResultShape().length;
            if (axes[1][i] < 0)
                axes[1][i] += b.getResultShape().length;

        }

        List<Integer> listA = new ArrayList<>();
        for (int i = 0; i < a.getResultShape().length; i++) {
            if (!Ints.contains(axes[0], i))
                listA.add(i);
        }

        int[] newAxesA = Ints.concat(Ints.toArray(listA), axes[0]);


        List<Integer> listB = new ArrayList<>();
        for (int i = 0; i < b.getResultShape().length; i++) {
            if (!Ints.contains(axes[1], i))
                listB.add(i);
        }

        int[] newAxesB = Ints.concat(axes[1], Ints.toArray(listB));

        int n2 = 1;
        int aLength = Math.min(a.getResultShape().length, axes[0].length);
        for (int i = 0; i < aLength; i++) {
            n2 *= a.getResultShape()[axes[0][i]];
        }

        //if listA and listB are empty these do not initialize.
        //so initializing with {1} which will then get overridden if not empty
        int[] newShapeA = {-1, n2};
        int[] oldShapeA;
        if (listA.size() == 0) {
            oldShapeA = new int[] {1};
        } else {
            oldShapeA = Ints.toArray(listA);
            for (int i = 0; i < oldShapeA.length; i++)
                oldShapeA[i] = a.getResultShape()[oldShapeA[i]];
        }

        int n3 = 1;
        int bNax = Math.min(b.getResultShape().length, axes[1].length);
        for (int i = 0; i < bNax; i++) {
            n3 *= b.getResultShape()[axes[1][i]];
        }


        int[] newShapeB = {n3, -1};
        int[] oldShapeB;
        if (listB.size() == 0) {
            oldShapeB = new int[] {1};
        } else {
            oldShapeB = Ints.toArray(listB);
            for (int i = 0; i < oldShapeB.length; i++)
                oldShapeB[i] = b.getResultShape()[oldShapeB[i]];
        }


        DifferentialFunction at = f()
                .reshape(f().permute
                        (a,newAxesA),newShapeA);
        DifferentialFunction bt = f()
                .reshape(f()
                        .permute(b,newAxesB),newShapeB);

        DifferentialFunction ret = f().mmul(at,bt);
        int[] aPlusB = Ints.concat(oldShapeA, oldShapeB);
        return f().reshape(ret,aPlusB);
    }


    public TensorMmul(INDArray x, INDArray y, int[][] axes) {
        super(x, y);
        this.axes = axes;
        this.extraArgs = new Object[] {axes};
    }

    /**
     * Initialize with the given
     * input, pairwise transform, result, and number
     * of elements
     *
     * @param x the input
     * @param y the pairwise transform
     * @param z the result
     */
    public TensorMmul(INDArray x, INDArray y, INDArray z, int[][] axes) {
        super(x, y, z, 0);
        this.axes = axes;
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public void exec() {
        if (this.z != null)
            this.z.assign(Nd4j.tensorMmul(x, y, z, axes));
        else
            this.z = Nd4j.tensorMmul(x, y, axes);
    }

    @Override
    public int opNum() {
        return 3;
    }

    @Override
    public long n() {
        return 0;
    }

    @Override
    public String name() {
        return "tensormmul";
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
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        TensorMmul that = (TensorMmul) o;

        if (addedEdges != that.addedEdges) return false;
        if (!Arrays.deepEquals(axes, that.axes)) return false;
        return mMulTranspose != null ? mMulTranspose.equals(that.mMulTranspose) : that.mMulTranspose == null;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + Arrays.deepHashCode(axes);
        result = 31 * result + (addedEdges ? 1 : 0);
        result = 31 * result + (mMulTranspose != null ? mMulTranspose.hashCode() : 0);
        return result;
    }
}
