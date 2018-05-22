package org.deeplearning4j.parallelism.main;

import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Creates an {@link MultiDataSetIterator}
 *
 * @author Adam Gibson
 */
public interface MultiDataSetProviderFactory {

    /**
     * Create an {@link MultiDataSetIterator}
     * @return
     */
    MultiDataSetIterator create();

}
