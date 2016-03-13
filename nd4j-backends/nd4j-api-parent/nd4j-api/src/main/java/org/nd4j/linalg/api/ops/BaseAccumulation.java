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

package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Base class for accumulation, initiates the initial entry
 * with respect to the child class. Also contains baseline fields
 * for the over all field with accumulation.
 *
 * @author Adam Gibson
 */
public abstract class BaseAccumulation extends BaseOp implements Accumulation {
    protected Number finalResult;
    protected IComplexNumber finalResultComplex;
    protected boolean applyFinalTransform = true;
    public BaseAccumulation() {
    }


    /**
     * Initialize with the given
     * input, pairwise transform, result, and number
     * of elements
     *
     * @param x the input
     * @param y the pairwise transform
     * @param z the result
     * @param n the number of elements
     */
    public BaseAccumulation(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
        init();
    }

    public BaseAccumulation(INDArray x, INDArray y, int n) {
        this(x, y, x, n);
    }

    public BaseAccumulation(INDArray x) {
        this(x, null, x, x.length());
    }

    public BaseAccumulation(INDArray x, INDArray y) {
        this(x, y, x, x.length());
    }

    private void init() {
        init(x, y, x, x.length());
    }

    @Override
    public INDArray noOp() {
        return x();
    }

    @Override
    public boolean applyFinalTransform() {
        return applyFinalTransform;
    }

    @Override
    public void setApplyFinalTransform(boolean applyFinalTransform) {
       this.applyFinalTransform = applyFinalTransform;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        numProcessed++;
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        numProcessed++;
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        numProcessed++;
        return origin;
    }

    @Override
    public float op(float origin, float other) {
        numProcessed++;
        return origin;
    }

    @Override
    public double op(double origin, double other) {
        numProcessed++;
        return origin;
    }

    @Override
    public double op(double origin) {
        numProcessed++;
        return origin;
    }

    @Override
    public float op(float origin) {
        numProcessed++;
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        numProcessed++;
        return origin;
    }

    @Override
    public double zeroDouble() {
        return 0.0;
    }

    @Override
    public float zeroFloat(){
        return 0.0f;
    }

    @Override
    public IComplexNumber zeroComplex() {
        return Nd4j.createComplexNumber(0.0,0.0);
    }

    @Override
    public int numProcessed() {
        return numProcessed;
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[]{zeroDouble()};
    }

    @Override
    public double combineSubResults(double first, double second){
        return update(first, second);
    }

    @Override
    public float combineSubResults(float first, float second){
        return update(first,second);
    }

    @Override
    public IComplexNumber combineSubResults(IComplexNumber first, IComplexNumber second) {
        return update(first,second);
    }

    @Override
    public double getAndSetFinalResult(double accum) {
        this.finalResult = accum;
        return accum;
    }

    @Override
    public float getAndSetFinalResult(float accum){
        this.finalResult = accum;
        return accum;
    }

    @Override
    public IComplexNumber getAndSetFinalResult(IComplexNumber accum){
        this.finalResultComplex = accum;
        return accum;
    }

    @Override
    public double calculateFinalResult(double accum, int n) {
        return accum;
    }

    @Override
    public float calculateFinalResult(float accum, int n) {
        return accum;
    }

    @Override
    public Number currentResult(){
        return finalResult;
    }

    @Override
    public void setFinalResult(Number number){
        this.finalResult = number;
    }

    @Override
    public void setFinalResultComplex(IComplexNumber number){
        this.finalResultComplex = number;
    }


    @Override
    public Number getFinalResult(){
        return finalResult;
    }

    @Override
    public IComplexNumber getFinalResultComplex(){
        return finalResultComplex;
    }

    @Override
    public void setZ(INDArray z) {
        super.setZ(z);
    }
}
