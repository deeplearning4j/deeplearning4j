package org.deeplearning4j.nn.api;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import java.io.Serializable;
import java.util.Map;

/**
 * Interface for a layer of a neural network.
 * This has an activation function, an input and output size,
 * weights, and a bias
 *
 * @author Adam Gibson
 */
public interface Layer extends Serializable,Cloneable,Model {


       /**
     * Parameter averaging
     * @param layer the layer to merge
     * @param batchSize the batch size to merge on
     */
    void merge(Layer layer,int batchSize);

    /**
     * Get the parameter
     * @param param the key of the parameter
     * @return the parameter vector/matrix with that particular key
     */
    INDArray getParam(String param);

    /**
     * Initialize the parameters
     */
    void initParams();

    /**
     * The param table
     * @return
     */
    Map<String,INDArray>  paramTable();

    void setParamTable(Map<String,INDArray> paramTable);


    /**
     * Set the parameter with a new ndarray
     * @param key the key to se t
     * @param val the new ndarray
     */
    void setParam(String key,INDArray val);

    INDArray activationMean();

    NeuralNetConfiguration conf();
    void setConfiguration(NeuralNetConfiguration conf);

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

    /**
     * Clone the layer
     * @return
     */
    Layer clone();


    /**
     * Propagate errors backwards for a particular layer
     * @param errors the errors to propagate
     */
    void backWard(INDArray errors);



}
