package org.nd4j.parameterserver.distributed.enums;

/**
 * @author raver119@gmail.com
 */
public enum TransportType {
    /**
     * This is default Transport implementation, suitable for network environments without UDP Broadcast support
     */
    ROUTED,

    /**
     * This implementation is suitable for network environments that DO support UDP Broadcast support.
     *
     * PLEASE NOTE: AWS/Azure instances do NOT support UDP Broadcast out of box
     */
    BROADCAST,

    /**
     * This option means you'll provide own Transport interface implementation via VoidParameterServer.init() method
     */
    CUSTOM,
}
