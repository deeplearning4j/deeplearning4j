package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.primitives.Pair;
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
    Pair<DataBuffer, long[]> createShapeInformation(int[] shape);

    /**
     * This method creates shapeInformation buffer, based on shape & order being passed in
     * @param shape
     * @return
     */
    Pair<DataBuffer, long[]> createShapeInformation(int[] shape, char order);

    /**
     * This method creates shapeInformation buffer, based on detailed shape information being passed in
     * @param shape
     * @return
     */
    Pair<DataBuffer, long[]> createShapeInformation(int[] shape, int[] stride, long offset, int elementWiseStride, char order);


    /**
     * This method creates shapeInformation buffer, based on detailed shape information being passed in
     * @param shape
     * @return
     */
    Pair<DataBuffer, long[]> createShapeInformation(int[] shape, int[] stride, long offset, int elementWiseStride, char order, long extras);

    /**
     * This method creates shapeInformation buffer, based on detailed shape information being passed in
     * @param shape
     * @return
     */
    Pair<DataBuffer, long[]> createShapeInformation(long[] shape, long[] stride, long offset, long elementWiseStride, char order, long extras);

    /**
     * This method creates long shapeInformation buffer, based on shape being passed in
     * @param shape
     * @return
     */
    Pair<DataBuffer, long[]> createShapeInformation(long[] shape);

    /**
     * This method creates long shapeInformation buffer, based on shape & order being passed in
     * @param shape
     * @return
     */
    Pair<DataBuffer, long[]> createShapeInformation(long[] shape, char order);

    /**
     * This method creates long shapeInformation buffer, based on detailed shape info being passed in
     * @param shape
     * @return
     */
    Pair<DataBuffer, long[]> createShapeInformation(long[] shape, long[] stride, long offset, long elementWiseStride, char order);



    /**
     * This method forces cache purge, if cache is available for specific implementation
     */
    void purgeCache();

    /**
     * This method returns memory used for cache, in bytes
     * @return
     */
    long getCachedBytes();
}
