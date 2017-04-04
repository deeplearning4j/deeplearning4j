package org.nd4j.autodiff;


public interface Ring<X> extends CommutativeGroup<X> {


    X mul(X i_v);


    X pow(int i_n);

}
