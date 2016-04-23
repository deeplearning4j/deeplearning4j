package org.nd4j.jita.flow;

import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.pointers.cuda.cudaStream_t;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.context.CudaContext;

/**
 * Interface describing flow controller.
 *
 * @author raver119@gmail.com
 */
public interface FlowController {

    void init(Allocator allocator);

    /**
     * This method ensures, that all asynchronous operations on referenced AllocationPoint are finished, and host memory state is up-to-date
     *
     * @param point
     */
    void synchronizeToHost(AllocationPoint point);

    /**
     * This method ensures, that all asynchronous operations on referenced AllocationPoint are finished
     * @param point
     */
    void waitTillFinished(AllocationPoint point);


    /**
     * This method is called after operation was executed
     *
     * @param result
     * @param operands
     */
    void registerAction(CudaContext context, INDArray result, INDArray... operands);

    /**
     * This method is called before operation was executed
     *
     * @param result
     * @param operands
     */
    CudaContext prepareAction(INDArray result, INDArray... operands);

    CudaContext prepareAction(AllocationPoint result, AllocationPoint... operands);

    void registerAction(CudaContext context, AllocationPoint result, AllocationPoint... operands);

    void waitTillReleased(AllocationPoint point);
}
