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

package org.nd4j.linalg.ops;


import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 * Baseline element wise operation so only applyTransformToOrigin has to be implemented.
 * This also handles the ability to perform scalar wise operations vs just
 * a functional transformation
 *
 * @author Adam Gibson
 */

public abstract class BaseElementWiseOp implements ElementWiseOp {

    protected INDArray from;
    //this is for operations like adding or multiplying a scalar over the from array
    protected Object scalarValue;
    protected Object[] extraArgs;


    /**
     * Apply the transformation at from[i]
     *
     * @param i the index of the element to apply the transform to
     */
    @Override
    public void applyTransformToOrigin(INDArray origin, int i) {
        if (origin instanceof IComplexNDArray) {
            IComplexNDArray c2 = (IComplexNDArray) origin;
            IComplexNumber transformed = apply(origin, getFromOrigin(origin, i), i);
            c2.putScalar(i, transformed);
        } else {
            Number f = apply(origin, getFromOrigin(origin, i), i);
            double val = f.doubleValue();
            if (Double.isNaN(val) || Double.isInfinite(val))
                val = Nd4j.EPS_THRESHOLD;
            if (origin.data().dataType() == (DataBuffer.FLOAT))
                origin.putScalar(i, val);
            else
                origin.putScalar(i, val);
        }

    }

    /**
     * Apply the transformation at from[i] using the supplied value
     *
     * @param origin       the origin ndarray
     * @param i            the index of the element to applyTransformToOrigin
     * @param valueToApply the value to apply to the given index
     */
    @Override
    public void applyTransformToOrigin(INDArray origin, int i, Object valueToApply) {
        if (valueToApply instanceof IComplexNumber) {
            if (origin instanceof IComplexNDArray) {
                IComplexNDArray c2 = (IComplexNDArray) origin;
                IComplexNumber apply = apply(origin, valueToApply, i);
                c2.putScalar(i, apply);
            } else
                throw new IllegalArgumentException("Unable to apply a non complex number to a real ndarray");
        } else {
            Number f = apply(origin, valueToApply, i);
            double val = f.doubleValue();
            if (Double.isNaN(val) || Double.isInfinite(val))
                val = Nd4j.EPS_THRESHOLD;
            if (origin.data().dataType() == (DataBuffer.FLOAT))
                origin.putScalar(i, val);
            else
                origin.putScalar(i, val);
        }


    }

    @Override
    public Object getFromOrigin(INDArray origin, int i) {
        if (origin instanceof IComplexNDArray) {
            IComplexNDArray c2 = (IComplexNDArray) origin;
            return c2.getComplex(i);
        }

        return origin.getDouble(i);
    }

    @Override
    public Object[] extraArgs() {
        return extraArgs;
    }

    @Override
    public void setExtraArgs(Object[] args) {
        this.extraArgs = args;
    }

    /**
     * The input matrix
     *
     * @return
     */
    @Override
    public INDArray from() {
        return from;
    }

    /**
     * Apply the transformation
     */
    @Override
    public void exec() {
        from.data().apply(this, from.offset());
    }
}