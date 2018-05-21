package org.nd4j.jita.allocator.pointers.cuda;

import lombok.NonNull;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.linalg.exception.ND4JException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

/**
 * @author raver119@gmail.com
 */
public class cudaStream_t extends CudaPointer {

    public cudaStream_t(@NonNull Pointer pointer) {
        super(pointer);
    }

    public int synchronize() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int res = nativeOps.streamSynchronize(this);
        if (res == 0)
            throw new ND4JException("CUDA exception happened. Terminating. Last op: [" + Nd4j.getExecutioner().getLastOp() +"]");

        return res;
    }
}
