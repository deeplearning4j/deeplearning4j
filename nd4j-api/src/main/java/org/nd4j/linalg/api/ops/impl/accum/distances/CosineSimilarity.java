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

package org.nd4j.linalg.api.ops.impl.accum.distances;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.accum.Norm2;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Cosine similarity
 * Note that you need to initialize
 * a scaling constant equal to the norm2 of the
 * vector
 *
 * @author Adam Gibson
 */
public class CosineSimilarity extends BaseAccumulation {
    private Number constantNormalizedByNorm2X, constantNormalizedByNorm2Y;

    public CosineSimilarity() {
    }

    public CosineSimilarity(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public CosineSimilarity(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public CosineSimilarity(INDArray x) {
        super(x);
    }

    public CosineSimilarity(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public void update(Number result) {
        currentResult = currentResult.doubleValue() + result.doubleValue();
        if (numProcessed() == n()) {
            currentResult = currentResult.doubleValue() / constantNormalizedByNorm2X.doubleValue() / constantNormalizedByNorm2Y.doubleValue();
        }

    }

    @Override
    public void update(IComplexNumber result) {
        currentComplexResult.addi(result);
        if (numProcessed() == n()) {
            currentComplexResult.set(currentComplexResult.realComponent().doubleValue() / constantNormalizedByNorm2X.doubleValue() / constantNormalizedByNorm2Y.doubleValue(), 0);
        }
    }

    @Override
    public String name() {
        return "cosinesimilarity";
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
        return (origin * other);
    }

    @Override
    public double op(double origin, double other) {
        numProcessed++;
        return origin * other;
    }


    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);
        if (y() != null)
            return new CosineSimilarity(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new CosineSimilarity(x.vectorAlongDimension(index, dimension));

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xForDimesnion = x.tensorAlongDimension(index, dimension);
        if (y() != null)
            return new CosineSimilarity(xForDimesnion, y.tensorAlongDimension(index, dimension), xForDimesnion.length());
        else
            return new CosineSimilarity(x.tensorAlongDimension(index, dimension));
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        this.constantNormalizedByNorm2X = Nd4j.getExecutioner().execAndReturn(new Norm2(x)).currentResult();
        this.constantNormalizedByNorm2Y = Nd4j.getExecutioner().execAndReturn(new Norm2(y)).currentResult();
        this.extraArgs = new Object[]{0.0,constantNormalizedByNorm2X, constantNormalizedByNorm2Y};
        this.initial = 0.0;
        this.initialComplex = Nd4j.createComplexNumber(0, 0);

    }
}
