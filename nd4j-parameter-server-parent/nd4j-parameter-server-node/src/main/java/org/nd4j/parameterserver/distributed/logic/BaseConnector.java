package org.nd4j.parameterserver.distributed.logic;

import io.aeron.Aeron;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.aeron.ipc.AeronConnectionInformation;
import org.nd4j.aeron.ipc.AeronNDArraySubscriber;
import org.nd4j.aeron.ipc.NDArrayCallback;
import org.nd4j.parameterserver.distributed.VoidParameterServer;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;

import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@Deprecated
public abstract class BaseConnector implements Connector {

    protected NodeRole nodeRole;
    protected Configuration configuration;


    protected Aeron.Context context;
    protected AtomicBoolean runner = new AtomicBoolean();
    protected NDArrayCallback callback;
    protected int streamId = 119;

    protected String localEndpoint;
    protected String remoteEndpoint;

    protected AeronNDArraySubscriber subscriber;

    public BaseConnector(@NonNull Configuration configuration, @NonNull NodeRole role) {
        this.nodeRole = role;
        this.configuration = configuration;

        if (context == null)
            context = new Aeron.Context();
    }

    public abstract void init();


    protected String getMatchedShardIp(Configuration configuration) {
        Set<String> locals = VoidParameterServer.getLocalAddresses();
        for (String ip: configuration.getShardAddresses()) {
            for (String localIp: locals) {
                if (ip.equals(localIp))
                    return ip;
            }
        }

        throw new IllegalStateException("No matches for shard IP");
    }
}
