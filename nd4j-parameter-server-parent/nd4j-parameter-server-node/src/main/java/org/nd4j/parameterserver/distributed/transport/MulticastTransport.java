package org.nd4j.parameterserver.distributed.transport;

import io.aeron.Aeron;
import io.aeron.FragmentAssembler;
import io.aeron.driver.MediaDriver;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.agrona.DirectBuffer;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.Clipboard;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;

/**
 * Transport implementation based on Aeron UDP multicast
 *
 * PLEASE NOTE: This transport will NOT work on AWS or Azure out of box, due to Amazon/Microsoft restrictions within their networks.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class MulticastTransport extends BaseTransport {
    protected String multicastChannelUri;

    public MulticastTransport() {
        // no-op
    }

    @Override
    public void init(@NonNull Configuration configuration, @NonNull Clipboard clipboard, @NonNull NodeRole role, @NonNull String localIp, short shardIndex) {
        if (configuration.getTtl() < 1)
            throw new ND4JIllegalStateException("For MulticastTransport you should have TTL >= 1, it won't work otherwise");

        if (configuration.getMulticastNetwork() == null || configuration.getMulticastNetwork().isEmpty())
            throw new ND4JIllegalStateException("For MulticastTransport you should provide IP from multicast network available/allowed in your environment, i.e.: 224.0.1.1");

        this.configuration = configuration;
        this.nodeRole = role;
        this.clipboard = clipboard;

        context = new Aeron.Context();

        driver = MediaDriver.launchEmbedded();

        context.aeronDirectoryName(driver.aeronDirectoryName());

        aeron = Aeron.connect(context);

        this.shardIndex = shardIndex;

        ip = "192.168.1.36";//localIp;

        multicastChannelUri = "aeron:udp?endpoint=" + configuration.getMulticastNetwork() + ":" + configuration.getMulticastPort();
        if (configuration.getMulticastInterface() != null && !configuration.getMulticastInterface().isEmpty())
            multicastChannelUri =  multicastChannelUri + "|interface=" + configuration.getMulticastInterface();

        multicastChannelUri = multicastChannelUri + "|ttl=" + configuration.getTtl();


        switch (nodeRole) {
            case BACKUP:
            case SHARD:
                /*
                    In case of Shard, unicast address for communication is known in advance
                 */
                unicastChannelUri = "aeron:udp?endpoint=" + ip + ":" + configuration.getUnicastPort();
                log.info("Shard unicast URI: {}/{}", unicastChannelUri, configuration.getStreamId());

                // this channel will be used to receive batches from Clients
                subscriptionForShards = aeron.addSubscription(unicastChannelUri, configuration.getStreamId());

                // this channel will be used to send completion reports back to Clients
                publicationForClients = aeron.addPublication(multicastChannelUri, configuration.getStreamId()+1);

                // this channel will be used for communication with other Shards
                publicationForShards = aeron.addPublication(multicastChannelUri, configuration.getStreamId() + 2);

                // this channel will be used to receive messages from other Shards
                subscriptionForClients = aeron.addSubscription(multicastChannelUri, configuration.getStreamId() + 2);

                messageHandlerForShards = new FragmentAssembler(
                        (buffer, offset, length, header) -> shardMessageHandler(buffer, offset, length, header)
                );

                messageHandlerForClients = new FragmentAssembler(
                        ((buffer, offset, length, header) -> internalMessageHandler(buffer, offset, length, header))
                );
                break;
            case CLIENT:

                /*
                    In case of Client, unicast will be one of shards, picked up with random
                 */
                unicastChannelUri = "aeron:udp?endpoint=" + ArrayUtil.getRandomElement(configuration.getShardAddresses()) + ":" + configuration.getUnicastPort();
                unicastChannelUri = "aeron:udp?endpoint=" + ip  + ":" + (configuration.getUnicastPort()) ;

                log.info("Client unicast URI: {}/{}", unicastChannelUri, configuration.getStreamId());

                /*
                 this channel will be used to send batches to Shards, it's 1:1 channel to one of the Shards
                */
                publicationForShards = aeron.addPublication(unicastChannelUri, configuration.getStreamId());

                // this channel will be used to receive completion reports from Shards
                subscriptionForClients = aeron.addSubscription(multicastChannelUri, configuration.getStreamId() + 1);

                messageHandlerForClients = new FragmentAssembler(
                        (buffer, offset, length, header) -> clientMessageHandler(buffer, offset, length, header)
                );
                break;
            default:
                log.warn("Unknown role passed: {}", nodeRole);
                throw new RuntimeException();
        }
    }

    /**
     * This command is possible to issue only from Client
     *
     * @param message
     */
    @Override
    protected void sendCommandToShard(VoidMessage message) {
        message.setTargetId(targetIndex);
        DirectBuffer buffer = message.asUnsafeBuffer();

        long result = publicationForShards.offer(buffer);

        if (result  < 0)
            for (int i = 0; i < 5 && result < 0; i++) {
                try {
                    // TODO: make this configurable
                    Thread.sleep(1000);
                } catch (Exception e) { }
                result = publicationForShards.offer(buffer);
            }

            // TODO: handle retransmit & backpressure separately

        if (result < 0)
            throw new RuntimeException("Unable to send message over the wire. Error code: " + result);
    }

    /**
     * This command is possible to issue only from Shard
     *
     * @param message
     */
    @Override
    protected void sendCoordinationCommand(VoidMessage message) {
        message.setTargetId((short) -1);
        publicationForShards.offer(message.asUnsafeBuffer());
    }

    /**
     * This command is possible to issue only from Shard
     *
     * @param message
     */
    @Override
    protected void sendFeedbackToClient(VoidMessage message) {
        message.setTargetId((short) -1);
        publicationForClients.offer(message.asUnsafeBuffer());
    }
}
