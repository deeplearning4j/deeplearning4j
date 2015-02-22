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

package org.nd4j.linalg.api.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.ArrayOps;
import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.util.Shape;

/**
 * Base activation function: mainly to
 * give the function a canonical representation
 * @author Adam Gibson
 */
public abstract class BaseActivationFunction implements ActivationFunction {
    /**
     * Name of the function
     *
     * @return the name of the function
     */
    @Override
    public String type() {
        return getClass().getName();
    }

    @Override
    public boolean equals(Object o) {
        return o.getClass().getName().equals(type());
    }

    /**
     * The type()
     *
     * @return
     */
    @Override
    public String toString() {
        return type();
    }


    /**
     * Returns the result of applying this function to {@code input}. This method is <i>generally
     * expected</i>, but not absolutely required, to have the following properties:
     * <p/>
     * <ul>
     * <li>Its execution does not cause any observable side effects.
     * <li>The computation is <i>consistent with equals</i>; that is, {@link Objects#equal
     * Objects.equal}{@code (a, b)} implies that {@code Objects.equal(function.apply(a),
     * function.apply(b))}.
     * </ul>
     *
     * @param input
     * @throws NullPointerException if {@code input} is null and this function does not accept null
     *                              arguments
     */
    @Override
    public INDArray apply(INDArray input) {
        INDArray passIn = input.dup();
        while (!Shape.shapeEquals(passIn.shape(), input.shape()))
            passIn = input.dup();

        ElementWiseOp op = new ArrayOps().from(passIn)
                .op(transformFactory())
                .build();

        passIn.data().apply(op);


        if (!Shape.shapeEquals(passIn.shape(), input.shape()))
            throw new IllegalStateException("Element wise operation of type " + op.toString() + " returned element not of same shape");

        return passIn;
    }


}
