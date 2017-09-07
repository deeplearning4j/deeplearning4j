package org.nd4j.autodiff.execution.conf;

/**
 * @author raver119@gmail.com
 */
public enum ExecutionMode {
    /**
     * all operations will be executed sequentially
     */
    SEQUENTIAL,

    /**
     * all operations will be following device id for execution mode selection
     */
    STRICT,

    /**
     * all operations that can be executed in parallel - will be executed in parallel
     */
    AUTO,
}
