package org.deeplearning4j.datasets.creator;

import org.deeplearning4j.datasets.iterator.DataSetIterator;

/**
 * Base interface for creating datasetiterators
 * @author Adam Gibson
 */
public interface DataSetIteratorFactory {
    public final static String NAME_SPACE = "org.deeplearning4j.datasets.creator";
    public final static String FACTORY_KEY = NAME_SPACE + ".datasetiteratorkey";
    /**
     * Create a dataset iterator
     * @return
     */
    DataSetIterator create();

}
