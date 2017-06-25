package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.Field;


public interface Differential<X extends Field<X>, D> {


    /**
     *
     * @param i_v
     * @return
     */
    D diff(Variable<X> i_v);
}
