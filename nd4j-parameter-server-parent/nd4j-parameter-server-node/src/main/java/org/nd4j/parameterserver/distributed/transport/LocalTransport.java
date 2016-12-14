package org.nd4j.parameterserver.distributed.transport;

import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;

/**
 *
 * @author raver119@gmail.com
 */
public class LocalTransport implements Transport {
    /**
     * This method does initialization of Transport instance
     *
     * @param configuration
     * @param role
     * @param localIp
     */
    @Override
    public void init(Configuration configuration, NodeRole role, String localIp) {

    }

    /**
     * This method accepts message for delivery, routing is applied according on message type
     *
     * @param message
     */
    @Override
    public void sendMessage(VoidMessage message) {

    }

    /**
     * This method accepts message from network
     *
     * @param message
     */
    @Override
    public void receiveMessage(VoidMessage message) {

    }

    /**
     * This method takes 1 message from "incoming messages" queue, blocking if queue is empty
     *
     * @return
     */
    @Override
    public VoidMessage takeMessage() {
        return null;
    }

    /**
     * This method starts transport mechanisms.
     * <p>
     * PLEASE NOTE: init() method should be called prior to launch() call
     *
     * @param threading
     */
    @Override
    public void launch(ThreadingModel threading) {

    }

    /**
     * This method stops transport system.
     */
    @Override
    public void shutdown() {

    }
}
