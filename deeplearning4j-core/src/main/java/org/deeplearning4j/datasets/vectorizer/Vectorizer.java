package org.deeplearning4j.datasets.vectorizer;


import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;

/**
 * A Vectorizer at its essence takes an input source
 * and converts it to a matrix for neural network consumption.
 * 
 * @author Adam Gibson
 *
 */
public interface Vectorizer extends Serializable {

	/**
	 * Vectorizes the input source in to a dataset
	 * @return Adam Gibson
	 */
	DataSet vectorize();



}
