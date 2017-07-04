package org.nd4j.parameterserver.distributed.conf;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.parameterserver.distributed.enums.ExecutionMode;
import org.nd4j.parameterserver.distributed.enums.FaultToleranceStrategy;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.enums.TransportType;

import java.io.Serializable;
import java.util.ArrayList;
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
public class VoidConfiguration implements Serializable {
    @Builder.Default
    private int streamId = 119;
    @Builder.Default
    private int unicastPort = 49876;
    @Builder.Default
    private int multicastPort = 59876;
    @Builder.Default
    private int numberOfShards = 1;;

    @Builder.Default
    private FaultToleranceStrategy faultToleranceStrategy = FaultToleranceStrategy.NONE;
    @Builder.Default
    private ExecutionMode executionMode = ExecutionMode.SHARDED;

    @Builder.Default
    private List<String> shardAddresses = new ArrayList<>();
    @Builder.Default
    private List<String> backupAddresses = new ArrayList<>();
    @Builder.Default
    private TransportType transportType = TransportType.ROUTED;

    // this is very important parameter
    private String networkMask;

    // This two values are optional, and have effect only for MulticastTransport
    @Builder.Default
    private String multicastNetwork = "224.0.1.1";
    private String multicastInterface;
    @Builder.Default
    private int ttl = 4;
    protected NodeRole forcedRole;

    // FIXME: probably worth moving somewhere else
    // this part is specific to w2v
    private boolean useHS = true;
    private boolean useNS = false;

    @Builder.Default
    private long retransmitTimeout = 1000;
    @Builder.Default
    private long responseTimeframe = 500;
    @Builder.Default
    private long responseTimeout = 30000;

    private String controllerAddress;

    public void setStreamId(int streamId) {
        if (streamId < 1)
            throw new ND4JIllegalStateException("You can't use streamId 0, please specify other one");

        this.streamId = streamId;
    }


    public void setShardAddresses(List<String> addresses) {
        this.shardAddresses = addresses;
    }

    public void setShardAddresses(String... ips) {
        if (shardAddresses == null)
            shardAddresses = new ArrayList<>();

        for (String ip : ips) {
            if (ip != null)
                shardAddresses.add(ip);
        }
    }

    public void setBackupAddresses(List<String> addresses) {
        this.backupAddresses = addresses;
    }

    public void setBackupAddresses(String... ips) {
        if (backupAddresses == null)
            backupAddresses = new ArrayList<>();

        for (String ip : ips) {
            if (ip != null)
                backupAddresses.add(ip);
        }
    }

    public void setExecutionMode(@NonNull ExecutionMode executionMode) {
        this.executionMode = executionMode;
    }
}
