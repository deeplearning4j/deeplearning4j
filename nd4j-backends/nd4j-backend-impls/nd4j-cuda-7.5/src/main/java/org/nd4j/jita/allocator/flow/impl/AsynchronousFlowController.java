package org.nd4j.jita.allocator.flow.impl;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaEvent_t;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.flow.FlowController;
import org.nd4j.jita.allocator.impl.AllocationPoint;
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

    }

    @Override
    public void waitTillFinished(AllocationPoint point) {

    }

    public void registerAction(INDArray result, INDArray... operands) {
        // no-op
        CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();

        cudaEvent_t event = new cudaEvent_t();

        JCuda.cudaEventCreate(event);
        JCuda.cudaEventRecord(event, context.getOldStream());


    }
}
