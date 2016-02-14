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

    public static AllocationShape buildAllocationShape(INDArray array) {
        AllocationShape shape = new AllocationShape();
        shape.setStride(array.elementWiseStride());
        shape.setOffset(array.offset());
        shape.setDataType(array.data().dataType());
        shape.setLength(array.length());

        return shape;
    }

    public static AllocationShape buildAllocationShape(BaseCudaDataBuffer array) {
        AllocationShape shape = new AllocationShape();
        shape.setStride(1);
        shape.setOffset(0);
        shape.setDataType(array.dataType());
        shape.setLength(array.length());

        return shape;
    }
}
