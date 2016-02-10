package org.nd4j.jita.balance.impl;

import lombok.NonNull;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.locks.Lock;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.balance.Balancer;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This is primitive balancer implementation, it accepts the first matching request without taking in account better candidates.
 * However, in exchange it's the fastest balancer, and suits workloads with many small memory regions.
 *
 * TODO: Balancer functionality should be merged into Mover
 * @author raver119@gmail.com
 */
@Deprecated
public class FirstInBalancer implements Balancer {
    private Configuration configuration;
    private CudaEnvironment environment;
    private Lock locker;


    private static Logger log = LoggerFactory.getLogger(FirstInBalancer.class);

    /**
     * This method initializes this Balancer instance
     *
     * @param configuration
     * @param environment
     */
    @Override
    public void init(@NonNull Configuration configuration, @NonNull CudaEnvironment environment, @NonNull Lock locker) {
        this.configuration = configuration;
        this.environment = environment;
        this.locker = locker;
    }

    /**
     * This method checks, if it's worth moving some memory region to device
     *
     * @param deviceId
     * @param point
     * @param shape
     * @return
     */
    @Override
    public AllocationStatus makePromoteDecision(Integer deviceId, AllocationPoint point, AllocationShape shape) {
        // TODO: to be decided on status here
       return null;
    }

    /**
     * This method checks, if it's worth moving some memory region to host.
     * For FirstInBalancer answer YES is constant answer, if memory usage is close to allocation threshold
     *
     * @param deviceId
     * @param point
     * @param shape
     * @return
     */
    @Override
    public AllocationStatus makeDemoteDecision(Integer deviceId, AllocationPoint point, AllocationShape shape) {
        if (!point.getAllocationStatus().equals(AllocationStatus.DEVICE))
            throw new IllegalStateException("You can't demote memory staged at ["+ point.getAllocationStatus()+"]");

        long maximumMemory = configuration.getMaximumDeviceAllocation();
        long allocatedMemory = environment.getAllocatedMemoryForDevice(deviceId);
        long currentLength = AllocationUtils.getRequiredMemory(shape);

        int singleDivider = 1;
        int allocDivider = 1;
        switch (configuration.getDeallocAggressiveness()) {
            case PEACEFUL:
                allocDivider = 3;
                singleDivider = 3;
                break;
            case REASONABLE:
                allocDivider = 4;
                singleDivider = 4;
                break;
            case URGENT:
                allocDivider = 10;
                singleDivider = 10;
                break;
            case IMMEDIATE:
                return AllocationStatus.ZERO;
            default:
                break;
        }

        if (currentLength > (maximumMemory / singleDivider) || allocatedMemory > (maximumMemory / allocDivider) ) {
            return AllocationStatus.ZERO;
        } else return AllocationStatus.DEVICE;
    }
}
