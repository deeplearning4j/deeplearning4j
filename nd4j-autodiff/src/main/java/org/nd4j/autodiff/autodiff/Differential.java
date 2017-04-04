package org.nd4j.autodiff.autodiff;

import org.nd4j.autodiff.Field;


public interface Differential<X extends Field<X>, D> {


    public D diff(Variable<X> i_v);
}
