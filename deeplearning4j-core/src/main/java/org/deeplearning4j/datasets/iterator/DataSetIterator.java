package org.deeplearning4j.datasets.iterator;

import java.io.Serializable;
import java.util.Iterator;

import org.deeplearning4j.datasets.DataSet;

/**
 * A DataSetIterator handles
 * traversing through a dataset and preparing
 * 
 * data for a neural network.
 * 
 * Typical usage of an iterator is akin to:
 * 
 * DataSetIterator iter = ..;
 * 
 * while(iter.hasNext()) {
 *     DataSet d = iter.next();
 *     //train network...
 * }
 * 
 * 
 * For custom numbers of examples/batch sizes you can call:
 * 
 * iter.next(num)
 * 
 * where num is the number of examples to fetch
 * 
 * 
 * @author Adam Gibson
 *
 */
public interface DataSetIterator extends Iterator<DataSet>,Serializable {

	/**
	 * Like the standard next method but allows a 
	 * customizable number of examples returned
	 * @param num the number of examples
	 * @return the next data set
	 */
	DataSet next(int num);
	
	int totalExamples();
	
	int inputColumns();
	
	int totalOutcomes();
	
	void reset();
	
	int batch();
	
	int cursor();
	
	int numExamples();
	
}
