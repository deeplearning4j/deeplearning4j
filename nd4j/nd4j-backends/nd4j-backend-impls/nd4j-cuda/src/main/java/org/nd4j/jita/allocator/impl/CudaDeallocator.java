package org.nd4j.jita.allocator.impl;

@Slf4j
public class CudaWorkspaceDeallocator implements Deallocator {

    private AllocationPoint point;
    private Map<Long, AllocationPoint> allocationsMap = new ConcurrentHashMap<>();

    public CudaDeallocator(@NonNull BaseCudaDataBuffer buffer,
                           @NonNull AllocationsMap allocationsMap) {
        this.point = buffer.getAllocationPoint();
        this.allocationsMap = allocationsMap;
    }

    @Override
    public void deallocate() {
        log.trace("Deallocating CUDA memory");
        // skipping any allocation that is coming from workspace
        /*if (point.isAttached()) {
            // TODO: remove allocation point as well?
            if (!allocationsMap.containsKey(point.getObjectId()))
                throw new RuntimeException();

            getFlowController().waitTillReleased(point);

            getFlowController().getEventsProvider().storeEvent(point.getLastWriteEvent());
            getFlowController().getEventsProvider().storeEvent(point.getLastReadEvent());

            allocationsMap.remove(point.getObjectId());

            return;
        }


        //log.info("Purging {} bytes...", AllocationUtils.getRequiredMemory(point.getShape()));
        if (point.getAllocationStatus() == AllocationStatus.HOST) {
            purgeZeroObject(point.getBucketId(), point.getObjectId(), point, false);
        } else if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
            purgeDeviceObject(0L, point.getDeviceId(), point.getObjectId(), point, false);

            // and we deallocate host memory, since object is dereferenced
            purgeZeroObject(point.getBucketId(), point.getObjectId(), point, false);
        }*/
    }
}