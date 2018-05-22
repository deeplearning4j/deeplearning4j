package org.nd4j.jita.allocator.enums;

/**
 * This enum describes possible memory allocation status/locations
 *
 * @author raver119@gmail.com
 */
public enum AllocationStatus {
    UNDEFINED, HOST, DEVICE, DELAYED, DEALLOCATED, CONSTANT,
}
