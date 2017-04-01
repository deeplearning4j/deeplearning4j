package org.nd4j.jita.allocator.enums;

/**
 * Three state trigger for memory access state representation
 *
 * @author raver119@gmail.com
 */
public enum AccessState {
    TICK, // region access started
    TACK, // regeion access finished
    TOE
}
