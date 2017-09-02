package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;

import java.util.List;


public interface Differential<X extends Field<ArrayField>, D> {


    String getFormula(List<Variable> variables);

    /**
     *
     * @param i_v
     * @return
     */
    List<D> diff(List<DifferentialFunction<ArrayField>> i_v);

}
