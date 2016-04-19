package org.nd4j.jita.allocator.pointers.cuda;

import org.nd4j.jita.allocator.pointers.CudaPointer;

/**
 * Created by raver119 on 19.04.16.
 */
public class cudaEvent_t extends CudaPointer{

    public cudaEvent_t( long address) {
        super(address);
    }
}
