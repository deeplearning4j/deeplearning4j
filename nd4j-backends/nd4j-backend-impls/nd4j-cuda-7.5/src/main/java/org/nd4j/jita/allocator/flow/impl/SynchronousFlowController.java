package org.nd4j.jita.allocator.flow.impl;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.flow.FlowController;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author raver119@gmail.com
 */
public class SynchronousFlowController implements FlowController {
    private static Logger log = LoggerFactory.getLogger(SynchronousFlowController.class);
    private volatile Allocator allocator;

    @Override
    public void init(Allocator allocator) {
        this.allocator = allocator;
    }

    @Override
    public void synchronizeToHost(AllocationPoint point) {
        CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();

        waitTillFinished(point);

        if (!point.isActualOnHostSide()) {

            //log.info("Synchronization started... " + point.getShape());

            // if this piece of memory is device-dependant, we'll also issue copyback once
            if (point.getAllocationStatus() == AllocationStatus.DEVICE && !point.isActualOnHostSide()) {
                context.syncOldStream();
                JCuda.cudaMemcpyAsync(
                        new Pointer(point.getHostPointer().address()),
                        new Pointer(point.getDevicePointer().address()),
                        AllocationUtils.getRequiredMemory(point.getShape()),
                        cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        context.getOldStream()
                );
            };

       //     context.syncOldStream();

            point.tickHostRead();
        }; // else log.info("Point is actual on host side! " + point.getShape());
    }

    @Override
    public void waitTillFinished(AllocationPoint point) {
        CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();
        context.syncOldStream();
    }
}
