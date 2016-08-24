package org.nd4j.jita.flow.impl;

import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * FlowController implementation suitable for CudaGridExecutioner
 *
 * Main difference here, is delayed execution support and forced execution trigger in special cases
 *
 * @author raver119@gmail.com
 */
public class GridFlowController extends SynchronousFlowController {

    /**
     * This method makes sure HOST memory contains latest data from GPU
     *
     * Additionally, this method checks, that there's no ops pending execution for this array
     *
     * @param point
     */
    @Override
    public void synchronizeToHost(AllocationPoint point) {
        if (!point.isConstant())
            if (Nd4j.getExecutioner() instanceof GridExecutioner)
                ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

        super.synchronizeToHost(point);
    }

    /**
     *
     * Additionally, this method checks, that there's no ops pending execution for this array
     * @param point
     */
    @Override
    public void waitTillFinished(AllocationPoint point) {
//        if (!point.isConstant())
//            if (Nd4j.getExecutioner() instanceof GridExecutioner)
//                ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

        super.waitTillFinished(point);
    }

    /**
     *
     * Additionally, this method checks, that there's no ops pending execution for this array
     *
     * @param point
     */
    @Override
    public void waitTillReleased(AllocationPoint point) {
//        if (!point.isConstant())
//            if (Nd4j.getExecutioner() instanceof GridExecutioner)
//                ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

        super.waitTillReleased(point);
    }
}
