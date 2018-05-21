package org.nd4j.linalg.api.memory.enums;

/**
 * @author raver119@gmail.com
 */
public enum LearningPolicy {
    /**
     * This policy means - we learn during 1 cycle,
     * and allocate workspace memory right after it's done.
     */
    FIRST_LOOP,

    /**
     * This policy means - we learn during multiple cycles,
     * and allocate after WorkspaceConfiguration.cyclesBeforeInitialization
     * or after manual call to MemoryWorkspace.initializeWorkspace
     */
    OVER_TIME,

    /**
     * This policy means - no learning is assumed, WorkspaceConfiguration.initialSize value will be primary determinant for workspace size
     */
    NONE,
}
