package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * @author raver119@gmail.com
 */
public interface ShapeInfoProvider {
    DataBuffer createShapeInformation(int[] shape, int[] stride, int offset, int elementWiseStride, char order);

    /**
     * This method forces cache purge, if cache is available for specific implementation
     */
    void purgeCache();
}
