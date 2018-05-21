package org.nd4j.jita.allocator.pointers.cuda;

import lombok.Getter;
import lombok.Setter;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.linalg.exception.ND4JException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author raver119@gmail.com
 */
public class cudaEvent_t extends CudaPointer {

    private AtomicBoolean destroyed = new AtomicBoolean(false);

    @Getter
    @Setter
    private long clock;

    @Getter
    @Setter
    private int laneId;

    @Getter
    @Setter
    private int deviceId;

    public cudaEvent_t(Pointer pointer) {
        super(pointer);
    }

    public boolean isDestroyed() {
        return destroyed.get();
    }

    public void markDestoryed() {
        destroyed.set(true);
    }

    public void destroy() {
        if (!isDestroyed()) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().destroyEvent(this);
            markDestoryed();
        }
    }

    public void synchronize() {
        if (!isDestroyed()) {
            int res = NativeOpsHolder.getInstance().getDeviceNativeOps().eventSynchronize(this);
            if (res == 0)
                throw new ND4JException("CUDA exception happened. Terminating. Last op: [" + Nd4j.getExecutioner().getLastOp() +"]");
        }
    }

    public void register(cudaStream_t stream) {
        if (!isDestroyed()) {
            int res = NativeOpsHolder.getInstance().getDeviceNativeOps().registerEvent(this, stream);
            if (res == 0)
                throw new ND4JException("CUDA exception happened. Terminating. Last op: [" + Nd4j.getExecutioner().getLastOp() +"]");
        }
    }
}
