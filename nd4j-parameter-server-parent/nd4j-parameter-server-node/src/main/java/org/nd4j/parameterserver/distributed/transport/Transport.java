package org.nd4j.parameterserver.distributed.transport;

import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.messages.MeaningfulMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;

/**
 * Transport interface describes Client -> Shard, Shard -> Shard, Shard -> Client communication
 *
 * @author raver119@gmail.com
 */
public interface Transport {
    enum ThreadingModel {
        SAME_THREAD, // DO NOT USE IT IN REAL ENVIRONMENT!!!11oneoneeleven
        SINGLE_THREAD, DEDICATED_THREADS,
    }

    void setIpAndPort(String ip, int port);

    String getIp();

    int getPort();

    short getShardIndex();


    short getTargetIndex();


    void addClient(String ip, int port);


    void addShard(String ip, int port);

    /**
     * This method does initialization of Transport instance
     *
     * @param voidConfiguration
     * @param role
     * @param localIp
     */
    void init(VoidConfiguration voidConfiguration, Clipboard clipboard, NodeRole role, String localIp, int localPort,
                    short shardIndex);


    /**
     * This method accepts message for delivery, routing is applied according on message opType
     *
     * @param message
     */
    void sendMessage(VoidMessage message);

    /**
     * This method accepts message for delivery, and blocks until response delivered
     *
     * @return
     */
    MeaningfulMessage sendMessageAndGetResponse(VoidMessage message);

    /**
     *
     * @param message
     */
    void sendMessageToAllShards(VoidMessage message);

    /**
     *
     * @param message
     */
    void sendMessageToAllClients(VoidMessage message, Long... exclusions);

    /**
     * This method accepts message from network
     *
     * @param message
     */
    void receiveMessage(VoidMessage message);

    /**
     * This method takes 1 message from "incoming messages" queue, blocking if queue is empty
     *
     * @return
     */
    VoidMessage takeMessage();

    /**
     * This method puts message into processing queue
     *
     * @param message
     */
    void putMessage(VoidMessage message);

    /**
     * This method peeks 1 message from "incoming messages" queue, returning null if queue is empty
     *
     * PLEASE NOTE: This method is suitable for debug purposes only
     *
     * @return
     */
    VoidMessage peekMessage();

    /**
     * This method starts transport mechanisms.
     *
     * PLEASE NOTE: init() method should be called prior to launch() call
     */
    void launch(ThreadingModel threading);

    /**
     * This method stops transport system.
     */
    void shutdown();

    /**
     * This method returns number of known Clients
     * @return
     */
    int numberOfKnownClients();

    /**
     * This method returns number of known Shards
     * @return
     */
    int numberOfKnownShards();

    /**
     * This method returns ID of this Transport instance
     * @return
     */
    long getOwnOriginatorId();
}
