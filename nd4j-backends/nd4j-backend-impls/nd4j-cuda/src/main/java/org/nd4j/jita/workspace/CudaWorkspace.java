package org.nd4j.jita.workspace;

import lombok.NonNull;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class CudaWorkspace extends Nd4jWorkspace {


    public CudaWorkspace(@NonNull WorkspaceConfiguration configuration) {
        super(configuration);
    }

    public CudaWorkspace(@NonNull WorkspaceConfiguration configuration, @NonNull String workspaceId) {
        super(configuration, workspaceId);
    }

    @Override
    protected void init() {
        super.init();

        if (currentSize.get() > 0) {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

            workspace.setHostPointer(new PagedPointer(nativeOps.mallocHost(currentSize.get(), 0)));
            workspace.setDevicePointer(new PagedPointer(nativeOps.mallocDevice(currentSize.get(), null, 0)));
        }
    }

    @Override
    public PagedPointer alloc(long requiredMemory, DataBuffer.Type type) {
        return this.alloc(requiredMemory, MemoryKind.DEVICE, type);
    }

    @Override
    public PagedPointer alloc(long requiredMemory, MemoryKind kind, DataBuffer.Type type) {
        return super.alloc(requiredMemory, kind, type);
    }
}
