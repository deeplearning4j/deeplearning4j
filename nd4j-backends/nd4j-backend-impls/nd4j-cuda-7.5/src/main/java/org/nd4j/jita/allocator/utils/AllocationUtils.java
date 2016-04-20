package org.nd4j.jita.allocator.utils;

import lombok.NonNull;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;


/**
 * @author raver119@gmail.com
 */
public class AllocationUtils {

    public static long getRequiredMemory(@NonNull AllocationShape shape) {
        return shape.getLength() * getElementSize(shape) ;
    }

    public static int getElementSize(@NonNull AllocationShape shape) {
        if (shape.getElementSize() > 0) return shape.getElementSize();
            else return (shape.getDataType() == DataBuffer.Type.DOUBLE ? 8 : 4);
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
        shape.setDataType(array.data().dataType());

        return shape;
    }

    /**
     * This method returns AllocationShape for the whole DataBuffer.
     *
     * @param buffer
     * @return
     */
    public static AllocationShape buildAllocationShape(DataBuffer buffer) {
        AllocationShape shape = new AllocationShape();
        shape.setStride(1);
        shape.setOffset(buffer.originalOffset());
        shape.setDataType(buffer.dataType());
        shape.setLength(buffer.length());

        return shape;
    }

    /**
     * This method returns AllocationShape for specific buffer, that takes in account its real shape: offset, length, etc
     *
     * @param buffer
     * @return
     */
    public static AllocationShape buildAllocationShape(JCudaBuffer buffer) {
        AllocationShape shape = new AllocationShape();
        shape.setStride(1);
        shape.setOffset(buffer.originalOffset());
        shape.setDataType(buffer.dataType());
        shape.setLength(buffer.length());

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
