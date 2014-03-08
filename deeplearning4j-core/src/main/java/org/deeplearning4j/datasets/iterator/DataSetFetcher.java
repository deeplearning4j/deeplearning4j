package org.deeplearning4j.datasets.iterator;

import java.io.Serializable;

import org.deeplearning4j.datasets.DataSet;

/**
 * A low level interface for loading datasets in to memory.
 * 
 * This is used by an {@link DataSetIterator} to
 * 
 * handle the specifics of loading data in to memory.
 * @author Adam Gibson
 *
 */
public interface DataSetFetcher extends Serializable {

	/**
	 * Whether the dataset has more to load
	 * @return whether the data set has more to load
	 */
	boolean hasMore();
	
	/**
	 * Returns the next data set
	 * @return the next dataset
	 */
	DataSet next();
	
	/**
	 * Fetches the next dataset. You need to call this
	 * to get a new dataset, otherwise {@link #next()}
	 * just returns the last data set fetch
	 * @param numExamples the number of examples to fetch
	 */
	void fetch(int numExamples);
	
	/**
	 * The number of labels for a dataset
	 * @return the number of labels for a dataset
	 */
	int totalOutcomes();
	/**
	 * The length of a feature vector for an individual example
	 * @return the length of a feature vector for an individual example
	 */
	int inputColumns();
	/**
	 * The total number of examples
	 * @return the total number of examples
	 */
	int totalExamples();
	/**
	 * Returns the fetcher back to the beginning of the dataset
	 */
	void reset();
	/**
	 * Direct access to a number represenative of iterating through a dataset
	 * @return a cursor similar to an index
	 */
	int cursor();
}
