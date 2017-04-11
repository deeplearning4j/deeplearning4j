package org.nd4j.autodiff;



public interface Field<X> extends CommutativeRing<X> {


    X inverse();


    X div(X i_v);

    double getReal();

    X[] args();

    default double getImaginary() {
        throw new UnsupportedOperationException();
    }
}
