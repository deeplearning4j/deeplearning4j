package org.nd4j.autodiff;


public interface Ring<X> extends CommutativeGroup<X> {

    X muli(X i_v);

    X muli(double v);

    X powi(int i_n);

    X mul(X i_v);

    X mul(double v);

    X pow(int i_n);

}
