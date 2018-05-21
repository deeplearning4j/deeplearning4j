package org.nd4j.linalg.api.memory.enums;

/**
 * @author raver119@gmail.com
 */
public enum AllocationPolicy {
    /**
     * This policy means - we're allocating exact values we specify for WorkspaceConfiguration.initialSize, or learn during loops
     */
    STRICT,

    /**
     * This policy means - we'll be overallocating memory, following WorkspaceConfiguration.overallocationLimit
     */
    OVERALLOCATE,
}
