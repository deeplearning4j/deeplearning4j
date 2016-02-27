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
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

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
        passThrough = true;
    }

    public CosineSimilarity(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
    }

    public CosineSimilarity(INDArray x, INDArray y, int n) {
        super(x, y, n);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
    }

    public CosineSimilarity(INDArray x) {
        super(x);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
    }

    public CosineSimilarity(INDArray x, INDArray y) {
        super(x, y);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
    }




    @Override
    public double update(double accum, double x){
        return accum + x;
    }

    @Override
    public double update(double accum, double x, double y){
        return accum + x * y;
    }

    @Override
    public float update(float accum, float x){
        return accum + x;
    }

    @Override
    public float update(float accum, float x, float y){
        return accum + x * y;
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x){
        return accum.add(x);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x, double y){
        return accum.add(x*y);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x){
        return accum.add(x);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x, IComplexNumber y) {
        return accum.add(x.mul(y));
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        return accum.add(x.mul(y));
    }

    @Override
    public int opNum() {
        return 2;
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
        CosineSimilarity ret;
        if (y() != null)
            ret = new CosineSimilarity(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            ret = new CosineSimilarity(x.vectorAlongDimension(index, dimension));
        ret.setApplyFinalTransform(applyFinalTransform());
        return ret;

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xForDimesnion = x.tensorAlongDimension(index, dimension);
        CosineSimilarity ret;
        if (y() != null)
            ret = new CosineSimilarity(xForDimesnion, y.tensorAlongDimension(index, dimension), xForDimesnion.length());
        else
            ret = new CosineSimilarity(x.tensorAlongDimension(index, dimension));
        ret.setApplyFinalTransform(applyFinalTransform());
        return ret;
    }

    @Override
    public void exec() {
        this.constantNormalizedByNorm2X = x.norm2Number();
        this.constantNormalizedByNorm2Y = y.norm2Number();
        this.extraArgs = new Object[]{0.0,constantNormalizedByNorm2X, constantNormalizedByNorm2Y};
        double dot = Nd4j.getBlasWrapper().dot(x,y);
        this.finalResult = dot / (constantNormalizedByNorm2X.doubleValue() * constantNormalizedByNorm2Y.doubleValue());
    }

    @Override
    public void exec(int... dimension){
        int[] retShape = ArrayUtil.removeIndex(x.shape(), dimension);
        int nOps = x.tensorssAlongDimension(dimension);
        z = Nd4j.create(retShape);
        for( int i = 0; i < nOps; i++ ){
            double d = Nd4j.getExecutioner().execAndReturn((CosineSimilarity) opForDimension(i,dimension)).getFinalResult().doubleValue();
            z.putScalar(i, d);
        }
    }

    @Override
    public double getAndSetFinalResult(double accum){
        if(applyFinalTransform()) {
            double d = accum / (constantNormalizedByNorm2X.doubleValue()*constantNormalizedByNorm2Y.doubleValue());
            this.finalResult = d;
            return d;
        }
        else {
            return accum;
        }

    }

    @Override
    public float getAndSetFinalResult(float accum){
        return (float) getAndSetFinalResult((double) accum);
    }

    @Override
    public IComplexNumber getAndSetFinalResult(IComplexNumber accum){
        finalResultComplex = Nd4j.createComplexNumber(accum.realComponent().doubleValue() / (constantNormalizedByNorm2X.doubleValue() * constantNormalizedByNorm2Y.doubleValue()), 0);
        return finalResultComplex;
    }

    @Override
    public double calculateFinalResult(double accum, int n){
        throw new UnsupportedOperationException("Not supported for passthrough op");
    }

    @Override
    public float calculateFinalResult(float accum, int n){
        throw new UnsupportedOperationException("Not supported for passthrough op");
    }
}
