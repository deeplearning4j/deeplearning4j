package org.nd4j.parameterserver.distributed.transport;

import io.aeron.Aeron;
import io.aeron.FragmentAssembler;
import io.aeron.Publication;
import io.aeron.Subscription;
import io.aeron.driver.MediaDriver;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.agrona.DirectBuffer;
import org.agrona.concurrent.UnsafeBuffer;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
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
    public void init(@NonNull Configuration configuration, @NonNull NodeRole role, @NonNull String localIp) {
        if (configuration.getTtl() < 1)
            throw new ND4JIllegalStateException("For MulticastTransport you should have TTL >= 1, it won't work otherwise");

        if (configuration.getMulticastNetwork() == null || configuration.getMulticastNetwork().isEmpty())
            throw new ND4JIllegalStateException("For MulticastTransport you should provide IP from multicast network available/allowed in your environment, i.e.: 224.0.1.1");

        this.configuration = configuration;
        this.nodeRole = role;

        driver = MediaDriver.launchEmbedded();

        context = new Aeron.Context();

        context.aeronDirectoryName(driver.aeronDirectoryName());

        aeron = Aeron.connect(context);

        ip = localIp;

        multicastChannelUri = "aeron:udp?endpoint=" + configuration.getMulticastNetwork() + ":" + configuration.getPort();
        if (configuration.getMulticastInterface() != null && !configuration.getMulticastInterface().isEmpty())
            multicastChannelUri =  multicastChannelUri + "|interface=" + configuration.getMulticastInterface();

        multicastChannelUri = multicastChannelUri + "|ttl=" + configuration.getTtl();


        switch (nodeRole) {
            case BACKUP:
            case SHARD:
                /*
                    In case of Shard, unicast address for communication is known in advance
                 */
                unicastChannelUri = "aeron:udp?endpoint=" + localIp + ":" + configuration.getPort();
                log.info("Shard unicast URI: {}", unicastChannelUri);

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
                unicastChannelUri = "aeron:udp?endpoint=" + ArrayUtil.getRandomElement(configuration.getShardAddresses()) + ":" + configuration.getPort();
                unicastChannelUri = "aeron:udp?endpoint=192.168.1.36:" + (configuration.getPort());

                log.info("Client unicast URI: {}", unicastChannelUri);

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
        long result = publicationForShards.offer(message.asUnsafeBuffer());

        log.info("offer result: {}", result);
    }

    /**
     * This command is possible to issue only from Shard
     *
     * @param message
     */
    @Override
    protected void sendCoordinationCommand(VoidMessage message) {
        publicationForShards.offer(message.asUnsafeBuffer());
    }

    /**
     * This command is possible to issue only from Shard
     *
     * @param message
     */
    @Override
    protected void sendFeedbackToClient(VoidMessage message) {
        publicationForClients.offer(message.asUnsafeBuffer());
    }
}
