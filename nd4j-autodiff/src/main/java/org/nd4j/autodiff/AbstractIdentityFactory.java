package org.nd4j.autodiff;

public interface AbstractIdentityFactory<X> {
    /**
     *
     * @return
     */
    X zero();


    /**
     *
     * @return
     */
    X one();

    /**
     * Scalar value
     * @param value
     * @return
     */
    X scalar(double value);

}
