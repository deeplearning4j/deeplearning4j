package org.nd4j.jita.allocator.pointers.cuda;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.pointers.CudaPointer;

/**
 * Created by raver119 on 19.04.16.
 */
public class cublasHandle_t extends CudaPointer {

    public cublasHandle_t(Pointer pointer) {
        super(pointer);
    }
}
