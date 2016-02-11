package org.nd4j.jita.balance;

import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;

/**
 * @author raver119@gmail.com
 */
@Deprecated
public interface Balancer {

    /**
     *
     * This method initializes this Balancer instance
     *
     * @param configuration
     * @param environment
     */
    void init(Configuration configuration, CudaEnvironment environment);

    /**
     * This method checks, if it's worth moving some memory region to device
     *
     * @param deviceId
     * @param point
     * @param shape
     * @return
     */
    AllocationStatus makePromoteDecision(Integer deviceId, AllocationPoint point, AllocationShape shape);

    /**
     * This method checks, if it's worth moving some memory region to host
     *
     * @param deviceId
     * @param point
     * @param shape
     * @return
     */
    AllocationStatus makeDemoteDecision(Integer deviceId, AllocationPoint point, AllocationShape shape);
}
