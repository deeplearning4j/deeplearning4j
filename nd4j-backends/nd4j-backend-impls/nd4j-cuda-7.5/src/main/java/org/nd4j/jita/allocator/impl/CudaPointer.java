package org.nd4j.jita.allocator.impl;

import org.bytedeco.javacpp.Pointer;

/**
 *
 * @author raver119@gmail.com
 */
public class CudaPointer extends Pointer {




    public CudaPointer(jcuda.Pointer pointer) {
        this.address = pointer.getNativePointer();
    }

    public CudaPointer(long address) {
        this.address = address;
    }

    public Pointer asPointer() {
        return (Pointer) this;
    }


}
