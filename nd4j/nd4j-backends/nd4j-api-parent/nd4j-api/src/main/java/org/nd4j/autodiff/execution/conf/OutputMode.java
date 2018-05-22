package org.nd4j.autodiff.execution.conf;

/**
 * @author raver119@gmail.com
 */
public enum OutputMode {
    /**
     * only final nodes of graph will be returned
     */
    IMPLICIT,

    /**
     * only declared output fields
     */
    EXPLICIT,

    /**
     * both options enabled
     */
    EXPLICIT_AND_IMPLICIT,

    /**
     * whole content of VariableSpace will be returned back
     */
    VARIABLE_SPACE,
}
