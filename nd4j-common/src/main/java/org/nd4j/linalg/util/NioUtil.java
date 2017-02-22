package org.nd4j.linalg.util;

import java.nio.*;

/**
 * NioUtils for operations on
 * nio buffers
 * @author Adam Gibson
 */
public class NioUtil {

    private NioUtil() {}

    public enum BufferType {
        INT, FLOAT, DOUBLE
    }

    /**
     * Copy from the given from buffer
     * to the to buffer at the specified
     * offsets and strides
     * @param n
     * @param bufferType
     * @param from the origin buffer
     * @param fromOffset the starting offset
     * @param fromStride the stride at which to copy from the origin
     * @param to the destination buffer
     * @param toOffset the starting point
     * @param toStride the to stride
     */
    public static void copyAtStride(int n, BufferType bufferType, ByteBuffer from, int fromOffset, int fromStride,
                    ByteBuffer to, int toOffset, int toStride) {
        // TODO: implement shape copy for cases where stride == 1
        ByteBuffer fromView = from;
        ByteBuffer toView = to;
        fromView.order(ByteOrder.nativeOrder());
        toView.order(ByteOrder.nativeOrder());
        switch (bufferType) {
            case INT:
                IntBuffer fromInt = fromView.asIntBuffer();
                IntBuffer toInt = toView.asIntBuffer();
                for (int i = 0; i < n; i++) {
                    int put = fromInt.get(fromOffset + i * fromStride);
                    toInt.put(toOffset + i * toStride, put);
                }
                break;
            case FLOAT:
                FloatBuffer fromFloat = fromView.asFloatBuffer();
                FloatBuffer toFloat = toView.asFloatBuffer();
                for (int i = 0; i < n; i++) {
                    float put = fromFloat.get(fromOffset + i * fromStride);
                    toFloat.put(toOffset + i * toStride, put);
                }
                break;
            case DOUBLE:
                DoubleBuffer fromDouble = fromView.asDoubleBuffer();
                DoubleBuffer toDouble = toView.asDoubleBuffer();
                for (int i = 0; i < n; i++) {
                    toDouble.put(toOffset + i * toStride, fromDouble.get(fromOffset + i * fromStride));

                }
                break;
            default:
                throw new IllegalArgumentException("Only floats and double supported");

        }


    }

}
