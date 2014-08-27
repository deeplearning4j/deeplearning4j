package org.deeplearning4j.nn.api;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.linalg.api.activation.ActivationFunction;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.layers.HiddenLayer;
import org.deeplearning4j.nn.WeightInit;

import java.io.Serializable;

/**
 * Created by agibsonccc on 8/26/14.
 */
public interface Layer extends Serializable,Cloneable {
    WeightInit getWeightInit();

    void setWeightInit(WeightInit weightInit);

    int getnIn();

    void setnIn(int nIn);

    int getnOut();

    void setnOut(int nOut);

    INDArray getW();

    void setW(INDArray w);

    INDArray getB();

    void setB(INDArray b);

    RandomGenerator getRng();

    void setRng(RandomGenerator rng);

    INDArray getInput();

    void setInput(INDArray input);

    ActivationFunction getActivationFunction();

    void setActivationFunction(
            ActivationFunction activationFunction);

    boolean isConcatBiases();

    void setConcatBiases(boolean concatBiases);


    public Layer clone();

    /**
     * Returns a transposed version of this hidden layer.
     * A transpose is just the bias and weights flipped
     * + number of ins and outs flipped
     * @return the transposed version of this hidden layer
     */
    HiddenLayer transpose();

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
}
