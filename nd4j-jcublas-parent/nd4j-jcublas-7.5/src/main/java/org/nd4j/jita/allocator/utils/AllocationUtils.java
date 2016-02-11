package org.nd4j.jita.allocator.utils;

import lombok.NonNull;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.linalg.api.buffer.DataBuffer;

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
}
