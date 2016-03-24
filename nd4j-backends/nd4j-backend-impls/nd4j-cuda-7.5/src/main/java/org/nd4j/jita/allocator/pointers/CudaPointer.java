package org.nd4j.jita.allocator.pointers;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class is simple logic-less holder for pointers derived from CUDA.
 *
 * PLEASE NOTE:
 * 1. All pointers are blind, and do NOT care about length/capacity/offsets/strides whatever
 * 2. They are really blind. Even data type is unknown.
 *
 * @author raver119@gmail.com
 */
public class CudaPointer extends Pointer {

    private static Logger logger = LoggerFactory.getLogger(CudaPointer.class);

    public CudaPointer(Pointer pointer, long capacity) {
        this.address = pointer.address();
        this.capacity = capacity;
        this.limit = capacity;
    }

    public CudaPointer(Pointer pointer, long capacity, long byteOffset) {
        this.address = pointer.address() + byteOffset;
        this.capacity = capacity;
        this.limit = capacity;
    }

    public CudaPointer(jcuda.Pointer pointer,  long capacity) {
        this.address = pointer.getNativePointer();
        this.capacity = capacity;
        this.limit = capacity;
        this.position = 0;
    }

    public CudaPointer(long address) {
        this.address = address;
    }

    public Pointer asNativePointer() {
        return new Pointer(this);
    }

    public FloatPointer asFloatPointer() {
        return new FloatPointer(this);
    }

    public DoublePointer asDoublePointer() {
        return new DoublePointer(this);
    }

    public IntPointer asIntPointer() {
        return new IntPointer(this);
    }

    public jcuda.Pointer asCudaPointer() {
        return new jcuda.Pointer(this.address());
    }

    /**
     * Returns 1 for Pointer or BytePointer else {@code Loader.sizeof(getClass())} or -1 on error.
     */
    @Override
    public int sizeof() {
        return 4;
    }
}
