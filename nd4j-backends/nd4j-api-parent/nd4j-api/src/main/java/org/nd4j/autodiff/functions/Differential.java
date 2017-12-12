package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.samediff.SDVariable;

import java.util.List;


public interface Differential {



    /**
     *
     * @param i_v
     * @return
     */
    List<SDVariable> diff(List<SDVariable> i_v);

}
