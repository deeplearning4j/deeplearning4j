package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.Field;

import java.util.List;


public interface Differential<X extends Field<X>, D> {


    String getFormula(List<Variable<X>> variables);

    /**
     *
     * @param i_v
     * @return
     */
    List<D> diff(List<DifferentialFunction<X>> i_v);

}
