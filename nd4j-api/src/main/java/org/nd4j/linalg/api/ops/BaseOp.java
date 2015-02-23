/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Base op. An op involves iterating over 2 buffers (x,y)  up to n elements
 * and applying a transform or accumulating a result.
 *
 * @author Adam Gibson
 */
public abstract class BaseOp implements Op {

    protected INDArray x,y;
    protected int n;

    /**
     * Base operation constructor
     * @param x the origin ndarray
     * @param y the pairwise ndarray
     * @param n the number of elements
     */
    public BaseOp(INDArray x, INDArray y, int n) {
        this.x = x;
        this.y = y;
        this.n = n;
    }

    /**
     * An op for one ndarray
     * @param x the ndarray
     */
    public BaseOp(INDArray x) {
        this(x,null,x.length());
    }


    @Override
    public INDArray x() {
        return x;
    }

    @Override
    public INDArray y() {
        return y;
    }



    @Override
    public int n() {
        return n;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return op(origin,other,null);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return op(origin,other,null);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return op(origin,other,null);
    }


    @Override
    public float op(float origin, float other) {
        return op(origin,other,null);
    }

    @Override
    public double op(double origin, double other) {
        return op(origin,other,null);
    }

    @Override
    public double op(double origin) {
        return op(origin,null);
    }

    @Override
    public float op(float origin) {
        return op(origin,null);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return op(origin, (Object[]) null);
    }
}
