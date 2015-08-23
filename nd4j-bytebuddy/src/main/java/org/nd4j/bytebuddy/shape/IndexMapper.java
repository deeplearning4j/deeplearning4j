package org.nd4j.bytebuddy.shape;

/**
 * Map an int array on to an index
 *
 * @author Adam Gibson
 */
public interface IndexMapper {

    /**
     * Map an index on to a set of coordinates
     * @param shape the shape of the matrix
     * @param index the index to map
     * @param numIndices the number of indices
     * @param ordering the ordering for the array
     * @return the coordinates
     */
    int[] ind2sub(int[] shape,int index,int numIndices,char ordering);

}
