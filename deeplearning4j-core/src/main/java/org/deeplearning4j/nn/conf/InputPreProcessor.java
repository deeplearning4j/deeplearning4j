package org.deeplearning4j.nn.conf;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Input pre processor used
 * for pre processing input before passing it
 * to the neural network.
 *
 * @author Adam Gibson
 */
public interface InputPreProcessor {


    /**
     * Pre process input for a multi layer network
     * @param input the input to pre process
     * @return the input to pre process
     */
    INDArray preProcess(INDArray input);

}
