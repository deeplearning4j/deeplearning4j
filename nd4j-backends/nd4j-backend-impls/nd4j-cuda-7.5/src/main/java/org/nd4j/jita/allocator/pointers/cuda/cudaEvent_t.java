package org.nd4j.jita.allocator.pointers.cuda;

import lombok.Getter;
import lombok.Setter;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author raver119@gmail.com
 */
public class cudaEvent_t extends CudaPointer{

    private AtomicBoolean destroyed = new AtomicBoolean(false);

    @Getter @Setter private long clock;

    @Getter @Setter private int laneId;

    public cudaEvent_t( long address) {
        super(address);
    }

    public boolean isDestroyed() {
        return destroyed.get();
    }

    public void markDestoryed() {
        destroyed.set(true);
    }

    public synchronized void desroy() {
        if (!isDestroyed()) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().destroyEvent(this.address());
            markDestoryed();
        }
    }

    public synchronized void synchronize() {
        if (!isDestroyed()) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().eventSynchronize(this.address());
        }
    }
}
