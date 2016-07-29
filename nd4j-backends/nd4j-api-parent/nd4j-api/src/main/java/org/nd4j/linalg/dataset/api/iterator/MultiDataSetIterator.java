package org.nd4j.linalg.dataset.api.iterator;

import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

import java.io.Serializable;
import java.util.Iterator;

/**An iterator for {@link org.nd4j.linalg.dataset.api.MultiDataSet} objects.
 * Typical usage is for machine learning algorithms with multiple independent input (features) and output (labels)
 * arrays.
 */
public interface MultiDataSetIterator extends Iterator<MultiDataSet>, Serializable {

    /** Fetch the next 'num' examples. Similar to the next method, but returns a specified number of examples
     *
     * @param num Number of examples to fetch
     */
    MultiDataSet next(int num);

    /** Set the preprocessor to be applied to each MultiDataSet, before each MultiDataSet is returned.
      * @param preProcessor MultiDataSetPreProcessor. May be null.
     */
    void setPreProcessor(MultiDataSetPreProcessor preProcessor);

    /**
     * Is resetting supported by this DataSetIterator? Many DataSetIterators do support resetting,
     * but some don't
     *
     * @return true if reset method is supported; false otherwise
     */
    boolean resetSupported();

    /**
     * Resets the iterator back to the beginning
     */
    void reset();

}
