package org.nd4j.autodiff.functions;

import org.nd4j.linalg.api.ops.impl.transforms.Variable;

import java.util.List;


public interface Differential {


    String getFormula(List<Variable> variables);

    /**
     *
     * @param i_v
     * @return
     */
    List<DifferentialFunction> diff(List<DifferentialFunction> i_v);

}
