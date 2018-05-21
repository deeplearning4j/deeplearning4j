package org.nd4j.parameterserver.distributed.transport;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.messages.MeaningfulMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;

/**
 *
 * @author raver119@gmail.com
 */
@Deprecated
public class LocalTransport implements Transport {
    /**
     * This method does initialization of Transport instance
     *
     * @param voidConfiguration
     * @param role
     * @param localIp
     */
    @Override
    public void init(VoidConfiguration voidConfiguration, Clipboard clipboard, NodeRole role, String localIp,
                    int localPort, short shardIndex) {

    }

    /**
     * This method accepts message for delivery, routing is applied according on message opType
     *
     * @param message
     */
    @Override
    public void sendMessage(VoidMessage message) {

    }

    @Override
    public int numberOfKnownClients() {
        return 0;
    }

    @Override
    public int numberOfKnownShards() {
        return 0;
    }

    /**
     * @param message
     */
    @Override
    public void sendMessageToAllShards(VoidMessage message) {

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
     * This method puts message into processing queue
     *
     * @param message
     */
    @Override
    public void putMessage(VoidMessage message) {

    }

    /**
     * This method peeks 1 message from "incoming messages" queue, returning null if queue is empty
     * <p>
     * PLEASE NOTE: This method is suitable for debug purposes only
     *
     * @return
     */
    @Override
    public VoidMessage peekMessage() {
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

    @Override
    public MeaningfulMessage sendMessageAndGetResponse(@NonNull VoidMessage message) {
        throw new UnsupportedOperationException();
    }

    @Override
    public short getShardIndex() {
        return 0;
    }

    @Override
    public short getTargetIndex() {
        return 0;
    }

    @Override
    public void setIpAndPort(String ip, int port) {

    }

    @Override
    public void addClient(String ip, int port) {
        //
    }

    @Override
    public String getIp() {
        return null;
    }

    @Override
    public int getPort() {
        return 0;
    }

    @Override
    public void addShard(String ip, int port) {
        // no-op
    }

    @Override
    public void sendMessageToAllClients(VoidMessage message, Long... exclusions) {
        // no-op
    }

    @Override
    public long getOwnOriginatorId() {
        return 0;
    }
}
