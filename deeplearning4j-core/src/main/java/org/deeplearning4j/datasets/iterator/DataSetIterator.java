package org.deeplearning4j.datasets.iterator;

import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;
import java.util.Iterator;



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
 *     //iterate network...
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
	 * @return the next data applyTransformToDestination
	 */
	DataSet next(int num);

    /**
     * Total examples in the iterator
     * @return
     */
	int totalExamples();

    /**
     * Input columns for the dataset
     * @return
     */
	int inputColumns();

    /**
     * The number of labels for the dataset
     * @return
     */
	int totalOutcomes();

    /**
     * Resets the iterator back to the beginning
     */
	void reset();

    /**
     * Batch size
     * @return
     */
	int batch();

    /**
     * The current cursor if applicable
     * @return
     */
	int cursor();

    /**
     * Total number of examples in the dataset
     * @return
     */
	int numExamples();


    /**
     * Set a pre processor
     * @param preProcessor a pre processor to set
     */
    void setPreProcessor(DataSetPreProcessor preProcessor);

	
}
