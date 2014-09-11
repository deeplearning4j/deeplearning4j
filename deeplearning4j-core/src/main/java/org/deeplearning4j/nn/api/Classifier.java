package org.deeplearning4j.nn.api;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;


/**
 * A classifier (this is for supervised learning)
 *
 * @author Adam Gibson
 */
public interface Classifier extends Model {


    /**
     * Assuming an input and labels are already set
     * will score based on what's already set
     * @return the f1 score for the already
     * set input/output
     */
    float score();

    /**
     * Sets the input and labels and returns a score for the prediction
     * wrt true labels
     * @param data the data to score
     * @return the score for the given input,label pairs
     */
    float score(DataSet data);

    /**
     * Returns the f1 score for the given examples.
     * Think of this to be like a percentage right.
     * The higher the number the more it got right.
     * This is on a scale from 0 to 1.
     * @param examples te the examples to classify (one example in each row)
     * @param labels the true labels
     * @return the scores for each ndarray
     */
    float score(INDArray examples, INDArray labels);

    /**
     * Returns the number of possible labels
     * @return the number of possible labels for this classifier
     */
    int numLabels();

    /**
     * Takes in a list of examples
     * For each row, returns a label
     * @param examples the examples to classify (one example in each row)
     * @return the labels for each example
     */
    int[] predict(INDArray examples);


    /**
     * Returns the probabilities for each label
     * for each example row wise
     * @param examples the examples to classify (one example in each row)
     * @return the likelihoods of each example and each label
     */
    INDArray labelProbabilities(INDArray examples);


    /**
     * Fit the model
     * @param examples the examples to classify (one example in each row)
     * @param labels the example labels(a binary outcome matrix)
     */
    void fit(INDArray examples,INDArray labels);

    /**
     * Fit the model
     * @param data the data to train on
     */
    void fit(DataSet data);


    /**
     * Fit the model
     * @param examples the examples to classify (one example in each row)
     * @param labels the example labels(a binary outcome matrix)
     * @param params extra parameters
     */
    void fit(INDArray examples,INDArray labels,Object[] params);

    /**
     * Fit the model
     * @param data the data to train on
     * @param params extra parameters
     */
    void fit(DataSet data,Object[] params);


    /**
     * Fit the model
     * @param examples the examples to classify (one example in each row)
     * @param labels the labels for each example (the number of labels must match
     *               the number of rows in the example
     */
    void fit(INDArray examples,int[] labels);


    /**
     * Fit the model
     * @param examples the examples to classify (one example in each row)
     * @param labels the labels for each example (the number of labels must match
     *               the number of rows in the example
     * @param params extra parameters
     */
    void fit(INDArray examples,int[] labels,Object[] params);

    /**
     * Iterate once on the model
     * @param examples the examples to classify (one example in each row)
     * @param labels the labels for each example (the number of labels must match
     *               the number of rows in the example
     * @param params extra parameters
     */
    void iterate(INDArray examples,int[] labels,Object[] params);



}
