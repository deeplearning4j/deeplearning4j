package org.deeplearning4j.nn.api;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import java.io.Serializable;

/**
 * Interface for a layer of a neural network.
 * This has an activation function, an input and output size,
 * weights, and a bias
 *
 * @author Adam Gibson
 */
public interface Layer extends Serializable,Cloneable {


    NeuralNetConfiguration conf();
    void setConfiguration(NeuralNetConfiguration conf);

    INDArray getW();

    void setW(INDArray w);

    INDArray getB();

    void setB(INDArray b);

     INDArray getInput();

    void setInput(INDArray input);


    INDArray preOutput(INDArray x);

    /**
     * Trigger an activation with the last specified input
     * @return the activation of the last specified input
     */
    INDArray activate();

    /**
     * Initialize the layer with the given input
     * and return the activation for this layer
     * given this input
     * @param input the input to use
     * @return
     */
    INDArray activate(INDArray input);

    /**
     * Return a transposed copy of the weights/bias
     * (this means reverse the number of inputs and outputs on the weights)
     *
     * @return the transposed layer
     */
    Layer transpose();

    Layer clone();
}
