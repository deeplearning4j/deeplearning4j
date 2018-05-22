package org.nd4j.jita.flow.impl;

import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * FlowController implementation suitable for CudaGridExecutioner
 *
 * Main difference here, is delayed execution support and forced execution trigger in special cases
 *
 * @author raver119@gmail.com
 */
public class GridFlowController extends SynchronousFlowController {

    private static Logger logger = LoggerFactory.getLogger(GridFlowController.class);

    /**
     * This method makes sure HOST memory contains latest data from GPU
     *
     * Additionally, this method checks, that there's no ops pending execution for this array
     *
     * @param point
     */
    @Override
    public void synchronizeToHost(AllocationPoint point) {
        if (!point.isConstant() && point.isEnqueued()) {
            waitTillFinished(point);
        }

        super.synchronizeToHost(point);
    }

    /**
     *
     * Additionally, this method checks, that there's no ops pending execution for this array
     * @param point
     */
    @Override
    public void waitTillFinished(AllocationPoint point) {
        if (!point.isConstant() && point.isEnqueued())
            Nd4j.getExecutioner().commit();

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
        /**
         * We don't really need special hook here, because if op is enqueued - it's still holding all arrays
         */

        super.waitTillReleased(point);
    }
}
