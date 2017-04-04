package org.nd4j.autodiff;


public interface ComplexNumber<R extends RealNumber<R>, X> extends Field<X> {


    X conjugate();


    R re();


    R im();
}
