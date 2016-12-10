package org.nd4j.parameterserver.distributed.transport;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;

/**
 * Transport interface describes Client -> Shard, Shard -> Shard, Shard -> Client communication
 *
 * @author raver119@gmail.com
 */
public interface Transport {


    void init(Configuration configuration, NodeRole role, String localIp);
}
