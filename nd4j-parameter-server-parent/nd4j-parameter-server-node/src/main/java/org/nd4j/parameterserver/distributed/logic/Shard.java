package org.nd4j.parameterserver.distributed.logic;

import lombok.NonNull;
import org.nd4j.aeron.ipc.AeronNDArraySubscriber;
import org.nd4j.parameterserver.ParameterServerListener;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;

/**
 * @author raver119@gmail.com
 */
@Deprecated
public class Shard extends BaseConnector {



    public Shard(@NonNull Configuration configuration) {
        super(configuration, NodeRole.SHARD);
    }

    public void init() {
        localEndpoint = getMatchedShardIp(configuration);


    }
}
