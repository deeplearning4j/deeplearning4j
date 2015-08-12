package org.nd4j.linalg.api.shape;

/**
 * @author Adam Gibson
 */
public interface Sub2Ind {

    /**
     * Map the given index
     * to the given sub
     * @param toMap the sub to map
     * @return
     */
    int map(int[] toMap);

}
