package org.nd4j.linalg.indexing;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An index in to a particular dimension.
 * This handles traversing indexes along a dimension
 * such as particular rows, or intervals.
 *
 * @author Adam Gibson
 */
public interface INDArrayIndex {
    /**
     * The ending for this index
     * @return
     */
    long end();

    /**
     * The start of this index
     * @return
     */
    long offset();

    /**
     * The total length of this index (end - start)
     * @return
     */
    long length();

    /**
     * The stride for the index (most of the time will be 1)
     * @return
     */
    long stride();


    /**
     * Return the current index
     * without incrementing the counter
     * @return
     */
    long current();

    /**
     * Returns true if there is another element
     * in the index to iterate over
     * otherwise false
     * @return
     */
    boolean hasNext();

    /**
     * Returns the next index
     * @return
     */
    long next();

    /**
     * Reverse the indexes
     */
    void reverse();

    /**
     * Returns true
     * if the index is an interval
     * @return
     */
    boolean isInterval();

    /**
     *
     * @param isInterval
     */
    void setInterval(boolean isInterval);

    /**
     * Init the index wrt
     * the dimension and the given nd array
     * @param arr the array to initialize on
     * @param begin the beginning index
     * @param dimension the dimension to initialize on
     */
    void init(INDArray arr, long begin, int dimension);

    /**
     * Init the index wrt
     * the dimension and the given nd array
     * @param arr the array to initialize on
     * @param dimension the dimension to initialize on
     */
    void init(INDArray arr, int dimension);

    void init(long begin, long end, long max);

    /**
     * Initiailize based on the specified begin and end
     * @param begin
     * @param end
     */
    void init(long begin, long end);

    void reset();
}
