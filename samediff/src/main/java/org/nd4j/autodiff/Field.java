package org.nd4j.autodiff;



public interface Field<X> extends CommutativeRing<X> {


    X inverse();


    X rdiv(X i_v);

    X div(X i_v);

    double getReal();

    X[] args();


    X rsub(double v);

    X rdiv(double v);
}
