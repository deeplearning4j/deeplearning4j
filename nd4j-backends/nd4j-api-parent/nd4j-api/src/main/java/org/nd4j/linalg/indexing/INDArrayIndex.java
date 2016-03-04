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
    int end();

    /**
     * The start of this index
     * @return
     */
    int offset();

    /**
     * The total length of this index (end - start)
     * @return
     */
    int length();

    /**
     * The stride for the index (most of the time will be 1)
     * @return
     */
    int stride();


    /**
     * Return the current index
     * without incrementing the counter
     * @return
     */
    int current();

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
    int next();

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
    void init(INDArray arr,int begin,int dimension);
    /**
     * Init the index wrt
     * the dimension and the given nd array
     * @param arr the array to initialize on
     * @param dimension the dimension to initialize on
     */
    void init(INDArray arr,int dimension);

    /**
     * Initiailize based on the specified begin and end
     * @param begin
     * @param end
     */
    void init(int begin,int end);

    void reset();
}
