package org.nd4j.autodiff;


public interface Group<X> {


    X negate();


    X plus(X i_v);


    X minus(X i_v);


    X mul(double i_n);
}
