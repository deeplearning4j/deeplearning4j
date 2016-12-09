package org.nd4j.parameterserver.distributed.transport;

import io.aeron.Aeron;
import io.aeron.Publication;
import io.aeron.Subscription;
import io.aeron.driver.MediaDriver;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;

/**
 * Transport implementation based on Aeron UDP multicast
 *
 * PLEASE NOTE: This transport will NOT work on AWS or Azure out of box, due to Amazon/Microsoft restrictions within their networks.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class MulticastTransport {

    private Configuration configuration;
    private NodeRole nodeRole;

    private Aeron aeron;
    private Aeron.Context context;

    private String channelUri;

    // TODO: move this to singleton holder
    private MediaDriver driver;

    private Publication publicationForShards;
    private Publication publicationForClients;

    private Subscription subscriptionForShards;
    private Subscription subscriptionForClients;

    public MulticastTransport() {
        // no-op
    }

    public void init(@NonNull Configuration configuration, @NonNull NodeRole role) {
        this.configuration = configuration;
        this.nodeRole = role;

        context = new Aeron.Context();

        driver = MediaDriver.launch();

        aeron = Aeron.connect(context);

        if (configuration.getMulticastNetwork() == null || configuration.getMulticastNetwork().isEmpty())
            throw new ND4JIllegalStateException("For MulticastTransport you should provide IP from multicast network available/allowed in your environment, i.e.: 224.0.1.1");

        channelUri = "aeron:udp?endpoint=" + configuration.getMulticastNetwork() + ":" + configuration.getPort() + "";
        if (configuration.getMulticastInterface() != null && !configuration.getMulticastInterface().isEmpty())
            channelUri =  channelUri + "|interface=" + configuration.getMulticastInterface();

        switch (nodeRole) {
            case SHARD:
                publicationForClients = aeron.addPublication(channelUri, configuration.getStreamId()+1);
                subscriptionForShards = aeron.addSubscription(channelUri, configuration.getStreamId());
                break;
            case CLIENT:
                publicationForClients = aeron.addPublication(channelUri, configuration.getStreamId());
                subscriptionForClients = aeron.addSubscription(channelUri, configuration.getStreamId() + 1);
                break;
            default:
                log.warn("Unknown role passed: {}", nodeRole);
        }

    }


}
