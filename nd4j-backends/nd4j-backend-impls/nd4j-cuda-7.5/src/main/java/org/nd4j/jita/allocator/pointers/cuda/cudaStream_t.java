package org.nd4j.jita.allocator.pointers.cuda;

import org.nd4j.jita.allocator.pointers.CudaPointer;

/**
 * Created by raver119 on 19.04.16.
 */
public class cudaStream_t extends CudaPointer {

    public cudaStream_t( long address) {
        super(address);
    }
}
