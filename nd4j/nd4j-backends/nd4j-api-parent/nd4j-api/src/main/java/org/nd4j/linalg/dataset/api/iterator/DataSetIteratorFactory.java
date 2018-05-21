package org.nd4j.linalg.dataset.api.iterator;

/**
 * Creates {@link DataSetIterator}.
 * Typically used in command line applications.
 *
 * @author Adam Gibson
 */
public interface DataSetIteratorFactory {
    /**
     *
     * @return
     */
    DataSetIterator create();

}
