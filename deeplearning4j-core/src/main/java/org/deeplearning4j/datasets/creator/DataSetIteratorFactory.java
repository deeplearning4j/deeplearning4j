package org.deeplearning4j.datasets.creator;

import org.deeplearning4j.datasets.iterator.DataSetIterator;

/**
 * Base interface for creating datasetiterators
 */
public interface DataSetIteratorFactory {

    DataSetIterator create();

}
