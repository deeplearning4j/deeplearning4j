package org.deeplearning4j.vectorizer;

import org.deeplearning4j.datasets.DataSet;


/**
 * Core api for vectorizing objects
 * @author Adam Gibson
 *
 */
public interface Vectorizer {

	/**
	 * Vectorizes an object.
	 * This makes the assumption that
	 * objects being vectorized have been
	 * injected via a constructor
	 * @return the vectorizer
	 */
	DataSet vectorize();
}
