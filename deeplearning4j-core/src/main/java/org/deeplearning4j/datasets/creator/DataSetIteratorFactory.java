package org.deeplearning4j.datasets.creator;

import org.deeplearning4j.datasets.iterator.DataSetIterator;

/**
 * Base interface for creating datasetiterators
 * @author Adam Gibson
 */
public interface DataSetIteratorFactory {
    /**
     * Create a dataset iterator
     * @return
     */
    DataSetIterator create();

}
