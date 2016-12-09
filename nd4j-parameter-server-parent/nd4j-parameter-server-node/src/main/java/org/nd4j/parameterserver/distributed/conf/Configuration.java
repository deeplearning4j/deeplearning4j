package org.nd4j.parameterserver.distributed.conf;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.parameterserver.distributed.enums.FaultToleranceStrategy;

import java.io.Serializable;
import java.util.List;

/**
 * Basic configuration pojo for VoidParameterServer
 * @author raver119@gmail.com
 */
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Slf4j
@Data
public class Configuration implements Serializable {
    private int streamId;
    private int port;
    private int numberOfShards;
    private FaultToleranceStrategy faultToleranceStrategy;
    private List<String> shardAddresses;
    private List<String> backupAddresses;

    // This two values are optional, and have effect only for MulticastTransport
    private String multicastNetwork;
    private String multicastInterface;
    private int ttl;

    public void setStreamId(int streamId) {
        if (streamId < 1 )
            throw new ND4JIllegalStateException("You can't use streamId 0, please specify other one");

        this.streamId = streamId;
    }
}
