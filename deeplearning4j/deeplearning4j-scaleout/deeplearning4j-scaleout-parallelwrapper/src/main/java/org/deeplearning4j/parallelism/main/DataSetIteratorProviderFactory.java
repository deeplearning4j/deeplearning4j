package org.deeplearning4j.parallelism.main;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Create a dataset iterator.
 * This is for use with {@link ParallelWrapperMain}
 *
 * @author Adam Gibson
 */
public interface DataSetIteratorProviderFactory {

    /**
     * Create an {@link DataSetIterator}
     * @return
     */
    DataSetIterator create();
}
