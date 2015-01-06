package org.deeplearning4j.nn.api;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A Model is meant for predicting something from data.
 * Note that this is not like supervised learning where
 * there are labels attached to the examples.
 *
 */
public interface Model {

    /**
     * All models have a fit method
     */
    void fit();

    /**
     * Perform one update  applying the gradient
     * @param gradient the gradient to apply
     */
    void update(Gradient gradient);

    /**
     * The score for the model
     * @return the score for the model
     */
    public double score();

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
     * the number of parameters for the model
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
     */
    void fit(INDArray data);


    /**
     * Run one iteration
     * @param input the input to iterate on
     */
    public void iterate(INDArray input);


    /**
     * Calculate a gradient
     * @return the gradient for this model
     */
    Gradient getGradient();

    /**
     * Get the gradient and score
     * @return the gradient and score
     */
    Pair<Gradient,Double> gradientAndScore();

    /**
     * The current inputs batch size
     * @return the current inputs batch size
     */
    public int batchSize();


    /**
     * The configuration for the neural network
     * @return the configuration for the neural network
     */
    NeuralNetConfiguration conf();

    /**
     * Setter for the configuration
     * @param conf
     */
    void setConf(NeuralNetConfiguration conf);

    /**
     * The input/feature matrix for the model
     * @return the input/feature matrix for the model
     */
    INDArray input();

    /**
     * Validate the input
     */
    void validateInput();

}
