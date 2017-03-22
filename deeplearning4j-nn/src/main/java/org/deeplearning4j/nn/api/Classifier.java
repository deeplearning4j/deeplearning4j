/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.api;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;


/**
 * A classifier (this is for supervised learning)
 *
 * @author Adam Gibson
 */
public interface Classifier extends Model {



    /**
     * Sets the input and labels and returns a score for the prediction
     * wrt true labels
     * @param data the data to score
     * @return the score for the given input,label pairs
     */
    double f1Score(DataSet data);

    /**
     * Returns the f1 score for the given examples.
     * Think of this to be like a percentage right.
     * The higher the number the more it got right.
     * This is on a scale from 0 to 1.
     * @param examples te the examples to classify (one example in each row)
     * @param labels the true labels
     * @return the scores for each ndarray
     */
    double f1Score(INDArray examples, INDArray labels);

    /**
     * Returns the number of possible labels
     * @return the number of possible labels for this classifier
     */
    int numLabels();

    /**
     * Train the model based on the datasetiterator
     * @param iter the iterator to train on
     */
    void fit(DataSetIterator iter);

    /**
     * Takes in a list of examples
     * For each row, returns a label
     * @param examples the examples to classify (one example in each row)
     * @return the labels for each example
     */
    int[] predict(INDArray examples);

    /**
     * Takes in a DataSet of examples
     * For each row, returns a label
     * @param dataSet the examples to classify
     * @return the labels for each example
     */
    List<String> predict(DataSet dataSet);

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
    void fit(INDArray examples, INDArray labels);

    /**
     * Fit the model
     * @param data the data to train on
     */
    void fit(DataSet data);



    /**
     * Fit the model
     * @param examples the examples to classify (one example in each row)
     * @param labels the labels for each example (the number of labels must match
     *               the number of rows in the example
     */
    void fit(INDArray examples, int[] labels);



}
