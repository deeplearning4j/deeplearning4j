package org.nd4j.autodiff.functions;

import java.util.List;


public interface Differential {



    /**
     *
     * @param i_v
     * @return
     */
    List<DifferentialFunction> diff(List<DifferentialFunction> i_v);

}
