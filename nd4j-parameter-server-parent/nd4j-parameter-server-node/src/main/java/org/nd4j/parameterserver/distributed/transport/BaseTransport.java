package org.nd4j.parameterserver.distributed.transport;

import io.aeron.Aeron;
import io.aeron.FragmentAssembler;
import io.aeron.Publication;
import io.aeron.Subscription;
import io.aeron.driver.MediaDriver;
import lombok.NonNull;
import org.agrona.concurrent.UnsafeBuffer;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;

import java.util.concurrent.LinkedBlockingQueue;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseTransport implements Transport {
    protected Configuration configuration;
    protected NodeRole nodeRole;

    protected Aeron aeron;
    protected Aeron.Context context;

    protected String unicastChannelUri;

    protected String ip;

    // TODO: move this to singleton holder
    protected MediaDriver driver;

    protected Publication publicationForShards;
    protected Publication publicationForClients;

    protected Subscription subscriptionForShards;
    protected Subscription subscriptionForClients;


    protected FragmentAssembler messageHandler;

    protected LinkedBlockingQueue<VoidMessage> messages = new LinkedBlockingQueue<>();

    @Override
    public void sendMessage(@NonNull VoidMessage message) {
        switch (message.getMessageType()) {
            // messages 0..9 inclusive are reserved for Client->Shard commands
            case 0:
            case 1:
            case 2:
            case 3:
            case 4:
                // TODO: check, if current role is Shard itself, in this case we want to modify command queue directly, to reduce network load
                // this command is possible to issue from any node role
                sendCommandToShard(message);
                break;
            // messages 10..19 inclusive are reserved for Shard->Clients commands
            case 10:
            case 11:
            case 12:
                // this command is possible to issue only from Shard
                sendFeedbackToClient(message);
                break;
            // messages 20..29 inclusive are reserved for Shard->Shard commands
            case 20:
            case 21:
            case 22:
                sendCoordinationCommand(message);
                break;
            default:
                throw new RuntimeException("Unknown messageType passed for delivery");
        }
    }

    /**
     * This method saves incoming message to the Queue, for later dispatch from higher-level code, like actual TrainingFunction or VoidParameterServer itself
     *
     * @param message
     */
    @Override
    public void receiveMessage(VoidMessage message) {
        try {
            messages.put(message);
        } catch (Exception e) {
            // do nothing
        }
    }

    /**
     * This method takes 1 message from "incoming messages" queue, blocking if queue is empty
     *
     * @return
     */
    @Override
    public VoidMessage takeMessage() {
        try {
            return messages.take();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public abstract void init(@NonNull Configuration configuration, @NonNull NodeRole role, @NonNull String localIp);

    /**
     * This command is possible to issue only from Client, but Client might be Shard or Backup at the same time
     *
     * @param message
     */
    protected abstract void sendCommandToShard(VoidMessage message);

    /**
     * This command is possible to issue only from Shard
     *
     * @param message
     */
    protected abstract void sendCoordinationCommand(VoidMessage message);

    /**
     * This command is possible to issue only from Shard
     * @param message
     */
    protected abstract void sendFeedbackToClient(VoidMessage message);
}
