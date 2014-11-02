package org.deeplearning4j.optimize.api;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Custom step function for line search
 *
 * @author Adam Gibson
 */
public interface StepFunction {

    /**
     * Step with the given parameters
     * @param x the current parameters
     * @param line the line to step
     * @param params
     */
    void step(INDArray x,INDArray line,Object[] params);


    /**
     * Step with no parameters
     */
    void step(INDArray x,INDArray line);


    void step();


}
