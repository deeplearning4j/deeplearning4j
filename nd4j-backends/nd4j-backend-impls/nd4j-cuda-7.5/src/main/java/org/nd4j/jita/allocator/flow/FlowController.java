package org.nd4j.jita.allocator.flow;

import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AllocationPoint;

/**
 * Interface describing flow controller.
 *
 * @author raver119@gmail.com
 */
public interface FlowController {

    void init(Allocator allocator);

    /**
     * This method ensures, that all asynchronous operations on referenced allocation point are finished, and host memory state is up-to-date
     *
     * @param point
     */
    void synchronizeToHost(AllocationPoint point);
}
