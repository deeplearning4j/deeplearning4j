package org.nd4j.linalg.api.ndarray;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * @author raver119@gmail.com
 */
public interface ShapeInfoProvider {
    /**
     * This method creates shapeInformation buffer, based on shape being passed in
     * @param shape
     * @return
     */
    Pair<DataBuffer, int[]> createShapeInformation(int[] shape);

    /**
     * This method creates shapeInformation buffer, based on shape & order being passed in
     * @param shape
     * @return
     */
    Pair<DataBuffer, int[]> createShapeInformation(int[] shape, char order);

    /**
     * This method creates shapeInformation buffer, based on detailed shape information being passed in
     * @param shape
     * @return
     */
    Pair<DataBuffer, int[]> createShapeInformation(int[] shape, int[] stride, long offset, int elementWiseStride, char order);

    /**
     * This method forces cache purge, if cache is available for specific implementation
     */
    void purgeCache();
}
