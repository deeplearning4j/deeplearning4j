package org.nd4j.jita.allocator.flow.impl;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaEvent_t;
import jcuda.runtime.cudaMemcpyKind;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.flow.FlowController;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.context.CudaContext;

/**
 * @author raver119@gmail.com
 */
public class AsynchronousFlowController implements FlowController{
    private volatile Allocator allocator;

    @Override
    public void init(Allocator allocator) {
        this.allocator = allocator;
    }

    @Override
    public void synchronizeToHost(AllocationPoint point) {
        if (!point.isActualOnHostSide()) {

            if (!point.isConstant())
                waitTillFinished(point);

            //  log.info("Synchronization started... " + point.getShape());

            // if this piece of memory is device-dependant, we'll also issue copyback once
            if (point.getAllocationStatus() == AllocationStatus.DEVICE && !point.isActualOnHostSide()) {
                CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();

                JCuda.cudaMemcpyAsync(
                        new Pointer(point.getHostPointer().address()),
                        new Pointer(point.getDevicePointer().address()),
                        AllocationUtils.getRequiredMemory(point.getShape()),
                        cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        context.getSpecialStream()
                );

                context.syncSpecialStream();
            }// else log.info("Not [DEVICE] memory, skipping...");


            // updating host read timer
            point.tickHostRead();
            //log.info("After sync... isActualOnHostSide: {}", point.isActualOnHostSide());
        }
    }

    @Override
    public void waitTillFinished(AllocationPoint point) {
        cudaEvent_t event = point.getLastEvent();
        if (event != null) {
            JCuda.cudaEventSynchronize(event);
            JCuda.cudaEventDestroy(event);
            point.setLastEvent(null);
        }
    }

    public void registerAction(INDArray result, INDArray... operands) {
        if (result == null) return;
        // no-op
        CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();

        cudaEvent_t event = new cudaEvent_t();

        JCuda.cudaEventCreate(event);
        JCuda.cudaEventRecord(event, context.getOldStream());

        AllocationPoint point = allocator.getAllocationPoint(result);
        point.setLastEvent(event);
        point.tickDeviceWrite();
    }
}
