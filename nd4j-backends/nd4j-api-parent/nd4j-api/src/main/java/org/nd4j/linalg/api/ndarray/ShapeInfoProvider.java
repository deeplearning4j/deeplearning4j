package org.nd4j.linalg.api.ndarray;

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
    DataBuffer createShapeInformation(int[] shape);

    /**
     * This method creates shapeInformation buffer, based on shape & order being passed in
     * @param shape
     * @return
     */
    DataBuffer createShapeInformation(int[] shape, char order);

    /**
     * This method creates shapeInformation buffer, based on detailed shape information being passed in
     * @param shape
     * @return
     */
    DataBuffer createShapeInformation(int[] shape, int[] stride, int offset, int elementWiseStride, char order);

    /**
     * This method forces cache purge, if cache is available for specific implementation
     */
    void purgeCache();
}
