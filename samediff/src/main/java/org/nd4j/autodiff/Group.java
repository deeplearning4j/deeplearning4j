package org.nd4j.autodiff;


public interface Group<X> {


    X negate();


    X add(X i_v);

    X add(double i_v);

    X sub(X i_v);


    X rsub(X i_v);

    X mul(double i_n);

    X sub(double i_v);
}
