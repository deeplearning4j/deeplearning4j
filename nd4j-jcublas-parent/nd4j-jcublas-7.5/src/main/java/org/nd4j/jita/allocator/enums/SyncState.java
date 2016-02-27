package org.nd4j.jita.allocator.enums;

/**
 * @author raver119@gmail.com
 */
public enum SyncState {
    UNDEFINED, // state is unknown
    DESYNC, // memory is desync
    SYNC, // memory is sync
    PARTIAL // memory is accessed using shapes, so patial sync/desync is possible
}
