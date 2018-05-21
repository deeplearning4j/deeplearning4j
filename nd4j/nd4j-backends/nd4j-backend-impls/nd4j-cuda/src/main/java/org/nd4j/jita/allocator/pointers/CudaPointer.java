package org.nd4j.jita.allocator.pointers;

import org.bytedeco.javacpp.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class is simple logic-less holder for pointers derived from CUDA.
 *
 * PLEASE NOTE:
 * 1. All pointers are blind, and do NOT care about length/capacity/offsets/strides whatever
 * 2. They are really blind. Even data opType is unknown.
 *
 * @author raver119@gmail.com
 */
public class CudaPointer extends Pointer {

    private static Logger logger = LoggerFactory.getLogger(CudaPointer.class);


    public CudaPointer(Pointer pointer) {
        this.address = pointer.address();
        this.capacity = pointer.capacity();
        this.limit = pointer.limit();
        this.position = pointer.position();
    }

    public CudaPointer(Pointer pointer, long capacity) {
        this.address = pointer.address();
        this.capacity = capacity;
        this.limit = capacity;
        this.position = 0;

        //   logger.info("Creating pointer: ["+this.address+"],  capacity: ["+this.capacity+"]");
    }

    public CudaPointer(Pointer pointer, long capacity, long byteOffset) {
        this.address = pointer.address() + byteOffset;
        this.capacity = capacity;
        this.limit = capacity;
        this.position = 0;
    }

    public CudaPointer(long address) {
        this.address = address;
    }

    public CudaPointer(long address, long capacity) {
        this.address = address;
        this.capacity = capacity;
        this.limit = capacity;
        this.position = 0;
    }

    public Pointer asNativePointer() {
        return new Pointer(this);
    }

    public FloatPointer asFloatPointer() {
        return new FloatPointer(this);
    }

    public LongPointer asLongPointer() {
        return new LongPointer(this);
    }

    public DoublePointer asDoublePointer() {
        return new DoublePointer(this);
    }

    public IntPointer asIntPointer() {
        return new IntPointer(this);
    }

    public ShortPointer asShortPointer() {
        return new ShortPointer(this);
    }

    public long getNativePointer() {
        return address();
    }

    /**
     * Returns 1 for Pointer or BytePointer else {@code Loader.sizeof(getClass())} or -1 on error.
     */
    @Override
    public int sizeof() {
        return 4;
    }
}
