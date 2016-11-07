package org.nd4j.jita.allocator.pointers.cuda;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

/**
 * Created by raver119 on 19.04.16.
 */
public class cudaStream_t extends CudaPointer {

    public cudaStream_t( Pointer pointer) {
        super(pointer);
    }

    public int synchronize() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        return nativeOps.streamSynchronize(this);
    }
}
