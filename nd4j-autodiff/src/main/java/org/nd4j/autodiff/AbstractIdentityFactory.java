package org.nd4j.autodiff;

public interface AbstractIdentityFactory<X> {
    /**
     *
     * @return
     * @param shape
     */
    X zero(int[] shape);


    /**
     *
     * @return
     * @param shape
     */
    X one(int[] shape);

    /**
     * Scalar value
     * @param value
     * @return
     */
    X scalar(double value);

}
