package org.deeplearning4j.nn.api;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A Model is meant for predicting something from data.
 * Note that this is not like supervised learning where
 * there are labels attached to the examples.
 *
 */
public interface Model {



    public float score();

    /**
     * Transform the data based on the model's output.
     * This can be anything from a number to reconstructions.
     * @param data the data to transform
     * @return the transformed data
     */
    INDArray transform(INDArray data);

    /**
     * Parameters of the model (if any)
     * @return the parameters of the model
     */
    INDArray params();

    /**
     * The number of parameters for the model
     * @return the number of parameters for the model
     * 
     */
    int numParams();

    /**
     * Set the parameters for this model.
     * This expects a linear ndarray which then be unpacked internally
     * relative to the expected ordering of the model
     * @param params the parameters for the model
     */
    void setParams(INDArray params);


    /**
     * Fit the model to the given data
     * @param data the data to fit the model to
     * @param params the params (mixed values)
     */
    void fit(INDArray data,Object[] params);

    /**
     * Fit the model to the given data
     * @param data the data to fit the model to
     */
    void fit(INDArray data);


    /**
     * Run one iteration
     * @param input the input to iterate on
     * @param params the extra params for the neural network(k, corruption level, max epochs,...)
     */
    public void iterate(INDArray input,Object[] params);


}
