package org.nd4j.jita.allocator.pointers;

import org.bytedeco.javacpp.Pointer;

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

    public CudaPointer(Pointer pointer) {
        this.address = pointer.address();
    }

    public CudaPointer(jcuda.Pointer pointer) {
        this.address = pointer.getNativePointer();
    }

    public CudaPointer(long address) {
        this.address = address;
    }

    public Pointer asNativePointer() {
        return (Pointer) this;
    }

    public jcuda.Pointer asCudaPointer() {
        return new jcuda.Pointer(this.address());
    }
}
