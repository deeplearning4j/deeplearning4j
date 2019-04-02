package org.nd4j.jita.allocator.impl;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.nd4j.linalg.api.memory.Deallocator;

@Slf4j
public class CudaDeallocator implements Deallocator {

    private AllocationPoint point;

    public CudaDeallocator(@NonNull BaseCudaDataBuffer buffer) {
        this.point = buffer.getAllocationPoint();
        if (this.point == null)
            throw new RuntimeException();
    }

    @Override
    public void deallocate() {
        log.trace("Deallocating CUDA memory");
        // skipping any allocation that is coming from workspace
        if (point.isAttached()) {
            // TODO: remove allocation point as well?
            if (!AtomicAllocator.getInstance().allocationsMap().containsKey(point.getObjectId()))
                throw new RuntimeException();

            AtomicAllocator.getInstance().getFlowController().waitTillReleased(point);

            AtomicAllocator.getInstance().getFlowController().getEventsProvider().storeEvent(point.getLastWriteEvent());
            AtomicAllocator.getInstance().getFlowController().getEventsProvider().storeEvent(point.getLastReadEvent());

            AtomicAllocator.getInstance().allocationsMap().remove(point.getObjectId());

            return;
        }


        //log.info("Purging {} bytes...", AllocationUtils.getRequiredMemory(point.getShape()));
        if (point.getAllocationStatus() == AllocationStatus.HOST) {
            AtomicAllocator.getInstance().purgeZeroObject(point.getBucketId(), point.getObjectId(), point, false);
        } else if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
            AtomicAllocator.getInstance().purgeDeviceObject(0L, point.getDeviceId(), point.getObjectId(), point, false);

            // and we deallocate host memory, since object is dereferenced
            AtomicAllocator.getInstance().purgeZeroObject(point.getBucketId(), point.getObjectId(), point, false);
        }
    }
}
