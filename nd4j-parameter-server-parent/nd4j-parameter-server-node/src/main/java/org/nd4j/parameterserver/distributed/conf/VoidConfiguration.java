package org.nd4j.parameterserver.distributed.conf;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.parameterserver.distributed.enums.ExecutionMode;
import org.nd4j.parameterserver.distributed.enums.FaultToleranceStrategy;
import org.nd4j.parameterserver.distributed.enums.NodeRole;

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
    private int streamId;
    private int unicastPort;
    private int multicastPort;
    private int numberOfShards;
    private FaultToleranceStrategy faultToleranceStrategy;
    private ExecutionMode executionMode;
    private List<String> shardAddresses = new ArrayList<>();
    private List<String> backupAddresses = new ArrayList<>();

    // this is very important parameter
    private String networkMask;

    // This two values are optional, and have effect only for MulticastTransport
    private String multicastNetwork;
    private String multicastInterface;
    private int ttl;
    protected NodeRole forcedRole;

    // FIXME: probably worth moving somewhere else
    // this part is specific to w2v
    private boolean useHS = true;
    private boolean useNS = false;

    private long retransmitTimeout;
    private long responseTimeframe;
    private long responseTimeout;

    public void setStreamId(int streamId) {
        if (streamId < 1)
            throw new ND4JIllegalStateException("You can't use streamId 0, please specify other one");

        this.streamId = streamId;
    }


    public void setShardAddresses(List<String> addresses) {
        this.shardAddresses = addresses;
    }

    public void setShardAddresses(String... Ips) {
        shardAddresses = new ArrayList<>();

        for (String ip : Ips) {
            if (ip != null)
                shardAddresses.add(ip);
        }
    }

    public void setBackupAddresses(List<String> addresses) {
        this.backupAddresses = addresses;
    }

    public void setBackupAddresses(String... Ips) {
        backupAddresses = new ArrayList<>();

        for (String ip : Ips) {
            if (ip != null)
                backupAddresses.add(ip);
        }
    }

    public void setExecutionMode(@NonNull ExecutionMode executionMode) {
        this.executionMode = executionMode;
    }

    public static class VoidConfigurationBuilder {
        private String multicastNetwork = "224.0.1.1";
        private int ttl = 4;
        private int streamId = 119;
        private int unicastPort = 49876;
        private int multicastPort = 59876;
        private int numberOfShards = 1;
        private FaultToleranceStrategy faultToleranceStrategy = FaultToleranceStrategy.NONE;
        private ExecutionMode executionMode = ExecutionMode.DISTRIBUTED;
        private long retransmitTimeout = 1000;
        private long responseTimeframe = 500;
        private long responseTimeout = 30000;
    }
}
