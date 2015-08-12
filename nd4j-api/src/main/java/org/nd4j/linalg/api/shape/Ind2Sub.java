package org.nd4j.linalg.api.shape;

/**
 * @author Adam Gibson
 */
public interface Ind2Sub {

    /**
     * Map the given index
     * to a sub
     * @param index the index to map
     * @return the map
     */
    int[] map(int index);

}
