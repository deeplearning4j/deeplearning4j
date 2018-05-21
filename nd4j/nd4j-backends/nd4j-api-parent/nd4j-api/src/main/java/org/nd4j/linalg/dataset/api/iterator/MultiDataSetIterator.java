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
     * Get the {@link MultiDataSetPreProcessor}, if one has previously been set.
     * Returns null if no preprocessor has been set
     *
     * @return Preprocessor
     */
    MultiDataSetPreProcessor getPreProcessor();

    /**
     * Is resetting supported by this DataSetIterator? Many DataSetIterators do support resetting,
     * but some don't
     *
     * @return true if reset method is supported; false otherwise
     */
    boolean resetSupported();

    /**
     * Does this MultiDataSetIterator support asynchronous prefetching of multiple MultiDataSet objects?
     * Most MultiDataSetIterators do, but in some cases it may not make sense to wrap this iterator in an
     * iterator that does asynchronous prefetching. For example, it would not make sense to use asynchronous
     * prefetching for the following types of iterators:
     * (a) Iterators that store their full contents in memory already
     * (b) Iterators that re-use features/labels arrays (as future next() calls will overwrite past contents)
     * (c) Iterators that already implement some level of asynchronous prefetching
     * (d) Iterators that may return different data depending on when the next() method is called
     *
     * @return true if asynchronous prefetching from this iterator is OK; false if asynchronous prefetching should not
     * be used with this iterator
     */
    boolean asyncSupported();

    /**
     * Resets the iterator back to the beginning
     */
    void reset();

}
