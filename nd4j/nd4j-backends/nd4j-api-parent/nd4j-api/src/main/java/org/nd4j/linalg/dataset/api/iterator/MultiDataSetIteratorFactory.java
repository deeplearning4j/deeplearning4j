package org.nd4j.linalg.dataset.api.iterator;

/**
 * Creates {@link MultiDataSetIterator}.
 * Typically used in command line applications.
 *
 * @author Adam Gibson
 */
public interface MultiDataSetIteratorFactory {


    /**
     * Create a {@link MultiDataSetIterator}
     * @return
     */
    MultiDataSetIterator create();

}
