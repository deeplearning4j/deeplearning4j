package org.deeplearning4j.models.sequencevectors.graph.enums;

/**
 * @author raver119@gmail.com
 */
public enum  NoEdgeHandling {
    SELF_LOOP_ON_DISCONNECTED,
    EXCEPTION_ON_DISCONNECTED,
    PADDING_ON_DISCONNECTED,
    CUTOFF_ON_DISCONNECTED,
    RESTART_ON_DISCONNECTED,
}
