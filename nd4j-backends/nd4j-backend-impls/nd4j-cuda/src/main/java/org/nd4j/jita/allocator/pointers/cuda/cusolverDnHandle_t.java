
package org.nd4j.jita.allocator.pointers.cuda;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.pointers.CudaPointer;

/**
 * Created by rcorbish 30-OCT-2016
 */
public class cusolverDnHandle_t extends CudaPointer {

    public cusolverDnHandle_t(Pointer pointer) {
        super(pointer);
    }
}
