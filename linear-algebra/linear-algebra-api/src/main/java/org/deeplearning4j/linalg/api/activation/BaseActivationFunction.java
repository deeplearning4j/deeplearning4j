package org.deeplearning4j.linalg.api.activation;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.ops.ArrayOps;
import org.deeplearning4j.linalg.ops.BaseElementWiseOp;

/**
 * Base activation function: mainly to give the function a canonical representation
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
        new ArrayOps().from(input.linearView())
                .op(transformClazz())
                .build().exec();
        return input;
    }



}
