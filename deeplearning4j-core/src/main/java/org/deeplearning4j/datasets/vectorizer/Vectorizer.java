package org.deeplearning4j.datasets.vectorizer;

import org.deeplearning4j.datasets.DataSet;

/**
 * A Vectorizer at its essence takes an input source
 * and converts it to a matrix for neural network consumption.
 * 
 * @author Adam Gibson
 *
 */
public interface Vectorizer {

	/**
	 * Vectorizes the input source in to a dataset
	 * @return Adam Gibson
	 */
	DataSet vectorize();



}
