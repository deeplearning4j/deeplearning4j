package org.nd4j.jita.allocator.flow.impl;

import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.flow.FlowController;
import org.nd4j.jita.allocator.impl.AllocationPoint;

/**
 * @author raver119@gmail.com
 */
public class AsynchronousFlowController implements FlowController{
    @Override
    public void init(Allocator allocator) {

    }

    @Override
    public void synchronizeToHost(AllocationPoint point) {

    }

    @Override
    public void waitTillFinished(AllocationPoint point) {

    }
}
