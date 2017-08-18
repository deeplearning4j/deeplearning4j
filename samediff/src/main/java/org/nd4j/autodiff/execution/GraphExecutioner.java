package org.nd4j.autodiff.execution;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.impl.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This interface
 * @author raver119@gmail.com
 */
public interface GraphExecutioner {
    enum Type {
        /**
         * Executor runs on the same box
         */
        LOCAL,

        /**
         * Executor runs somewhere else
         */
        REMOTE,
    }

    /**
     * This method returns Type of this executioner
     *
     * @return
     */
    Type getExecutionerType();

    /**
     * This method executes given graph and returns results
     *
     * @param graph
     * @return
     */
    INDArray[] executeGraph(SameDiff graph);

    /**
     * This method executes
     * @param id
     * @param variables
     * @return
     */
    INDArray[] executeGraph(int id, SDVariable... variables);


    /**
     * This method stores given graph for future execution
     *
     * @param graph
     * @return
     */
    int registerGraph(SameDiff graph);
}
