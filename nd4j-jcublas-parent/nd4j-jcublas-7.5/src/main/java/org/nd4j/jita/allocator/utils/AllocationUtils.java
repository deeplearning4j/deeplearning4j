package org.nd4j.jita.allocator.utils;

import lombok.NonNull;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;

/**
 * @author raver119@gmail.com
 */
public class AllocationUtils {

    public static long getRequiredMemory(@NonNull AllocationShape shape) {

        return shape.getLength() * getElementSize(shape) ;
    }

    public static int getElementSize(@NonNull AllocationShape shape) {
        return (shape.getDataType() == DataBuffer.Type.DOUBLE ? 8 : 4);
    }

    /**
     * This method returns AllocationShape for specific array, that takes in account its real shape: offset, length, etc
     *
     * @param array
     * @return
     */
    public static AllocationShape buildAllocationShape(INDArray array) {
        AllocationShape shape = new AllocationShape();
        shape.setStride(array.elementWiseStride());
        shape.setOffset(array.originalOffset());
        shape.setDataType(array.data().dataType());
        shape.setLength(array.length());

        return shape;
    }

    /**
     * This method returns AllocationShape for the whole DataBuffer.
     *
     * @param array
     * @return
     */
    public static AllocationShape buildAllocationShape(DataBuffer array) {
        AllocationShape shape = new AllocationShape();
        shape.setStride(1);
        shape.setOffset(0);
        shape.setDataType(array.dataType());
        shape.setLength(array.length());

        return shape;
    }

    /**
     * This method returns byte offset based on AllocationShape
     *
     * @return
     */
    public static long getByteOffset(AllocationShape shape) {
        return shape.getOffset() * getElementSize(shape);
    }
}
